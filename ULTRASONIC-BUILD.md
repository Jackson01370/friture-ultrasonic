# このリポジトリについて（超音波対応ビルド）

これは **Friture の改造版**であって、素の Friture ではありません。UltraMic 250K
を 250 kHz で回して 0〜125 kHz を見る／聞くための専用ビルドです。

**普通のマイクは使えません。** `SAMPLING_RATE` が 250000 に固定してあるためです。
通常の音声解析には Microsoft Store 版の Friture が別途入っているので、そちらを
使ってください（AppUserModelID が別なので共存します）。

---

## 起動

```
friture.bat
```

またはデスクトップ／スタートメニューの **Friture** ショートカット。

`python main.py` を直接叩くと**起動しません**。この PyQt5 が Qt のプラグイン
パスを自己登録しないため、`friture.bat` が `QT_PLUGIN_PATH` と
`QML2_IMPORT_PATH` を設定しています。

ショートカットの作り直し:

```
powershell -ExecutionPolicy Bypass -File scripts\make_shortcut.ps1 -StartMenu
```

---

## 実行環境

専用の venv はなく、**ultraScan の venv に相乗り**しています
（`..\ultraScan\.venv`）。`friture.bat` と `make_shortcut.ps1` はこの順で
インタプリタを探します:

1. `%FRITURE_PYTHON%`
2. `friture-master\.venv`（作れば自動的に優先される）
3. `..\ultraScan\.venv` ← 現在これ
4. PATH 上の python

改造の過程で ultraScan の venv に追加したもの:
`rtmixer` `pa-ringbuffer` `docutils` `platformdirs` `pyrr` `cython` `soxr`
（`soxr` は 250k→48k 変換に必要で、`pyproject.toml` の依存にも追加済み）

**Cython 拡張はビルドされていません。** この PC に C++ コンパイラが無いためで、
代わりに `friture_extensions/*.py` に numpy 実装を置いてあります。Python は
拡張モジュールを同名の `.py` より優先して読むので、あとから
`python setup.py build_ext --inplace` すれば**何も変更せずに**コンパイル版へ
切り替わります。

---

## 主な変更点

| 領域 | 内容 |
|---|---|
| 収録 | WASAPI 排他モード・モノラル・250 kHz 固定。他の host API は明示的に拒否 |
| 出力 | 48 kHz 固定。`friture/listen/playout.py` で soxr 変換 |
| 帯域リスニング | `friture/listen/` 一式。クリック／ドラッグで帯域選択、バンドパスとヘテロダイン |
| AGC | `friture/listen/agc.py`。ultraScan の `AGCGain` 移植 |
| ノイズ低減 | `friture/listen/denoise.py` + `spectral_background.py` |
| ノイズゲート | `friture/listen/gate.py` |
| リミッター | `friture/listen/limiter.py`。ゲイン上限 150 dB に対応するため |
| 表示 | 対数目盛り既定、上限 125 kHz、FFT 16384、A特性オフ、背景減算オプション |
| オクターブ帯域 | `NOCTAVE` 9→13、係数を 250 kHz で再生成（39バンド 12.4 Hz〜80.6 kHz） |

設計理由はすべて各ファイルの docstring に書いてあります。特に
`friture/listen/band_dsp.py` の冒頭は、なぜ DDC 構造なのか（ドラッグ中に音を
切らないため）を説明しています。

---

## 検証のしかた

この PC では GUI を普通に起動して確認するのが難しい場面が多いので、
`scripts/verify/` に確認用スクリプトを置いてあります。

```
python -m unittest friture.test.test_listen_band friture.test.test_extension_fallbacks
                                              # 128 件（GUI 不要）

python scripts\verify\qml_and_interaction.py  # QML の読み込み・クリック・ドラッグ 16 項目
python scripts\verify\control_bar_layout.py   # バーの高さとプロット領域の実測
python scripts\verify\pipeline_smoke.py       # 音声パイプライン全段（GUI 不要）

python scripts\verify\capture_is_really_wideband.py  # ★ 収録が本物か
python scripts\verify\noise_survey.py                # 雑音の性質を測る
python scripts\verify\end_to_end.py                  # 実収録→聴取まで通し
python scripts\verify\tap_count_tradeoff.py          # タップ数と遅延・CPU の関係
```

`capture_is_really_wideband.py` は特に重要です。MME と DirectSound は 250 kHz
要求を「成功」で受けたうえで **48 kHz をアップサンプルした偽データ**を返します。
30 kHz 以上のエネルギーがゼロかどうかで判別できます。

---

## 実測値（判断の根拠）

| 項目 | 値 |
|---|---|
| 静穏時の広帯域 rms | −40 dBFS 前後 |
| 24994 Hz の狭帯域干渉 | 周囲より **+11.9 dB**（害虫忌避器） |
| 95734 Hz の干渉 | 周囲より +14.2 dB |
| ノイズフロアの定常性 | 4 秒の前後半で 0.1 dB 未満 |
| 40–50 kHz 帯（ヘテロダイン後） | −73 dBFS 前後 |
| 快適に聞ける手動ゲイン | +60 dB 前後（+75 dB で 43% クリップ） |
| 1ブロック(8.2ms)あたりの処理 | 帯域 0.4 / ノイズ低減 0.42 / リミッター 0.56 ms |

---

## 分かっているハマりどころ

**マイクを掴めるのは1プロセスだけ。** WASAPI 排他モードなので、Friture や
ultraScan、終了し損ねた `pythonw` が残っていると次は
`Invalid device [PaErrorCode -9996]` で失敗します。マイクの故障に見えますが
先客がいるだけです。

```powershell
Get-Process pythonw,python | Where-Object { $_.Path -like "*ultraScan*" }
```

**バンドパスは 24 kHz 超では無音です。** 再生が 48 kHz なので物理的に出せません。
バーに警告が出ます。超音波はヘテロダインで。

---

## 未着手・見送った項目

- **帯域リスニングの設定がセッション間で保存されない**（起動ごとに 45 kHz / 10 kHz
  に戻る）。スペクトログラムの背景減算は保存されます
- **`Listen` オン時のバーが 3 行（132 px）になる**幅がある。ゲインやノイズの項目を
  設定ダイアログへ移せば詰められる（提案済み・保留）
- **Cython 拡張が未ビルド**（上記）
- **friture 専用の venv が無い**（上記）
- ライブ監視は入出力のクロック差を補正していないので、数分に一度ドリフトによる
  クリックが入る
