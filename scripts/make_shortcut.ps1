<#
.SYNOPSIS
    Create a desktop (and optionally Start-menu) shortcut that launches this
    Friture build like an app.

.DESCRIPTION
    The .lnk points straight at pythonw.exe running scripts/friture_app.py, so
    there is no console flash at all -- friture.bat still works from a
    terminal, it just blinks a cmd window on its way through.

    The shortcut carries:

      - resources/images/friture.ico
      - System.AppUserModel.ID = the same id the process claims at startup
        (friture/analyzer.py sets it with SetCurrentProcessExplicitAppUserModelID)

    That last property is the whole reason this is a script and not three
    lines of WScript.Shell. Without a matching id Windows treats the pinned
    shortcut and the running window as two different apps: the pinned icon
    never lights up and a second button appears beside it. WScript.Shell
    cannot write it -- that needs IPropertyStore -- hence the C# helper below.

    Adapted from the same script in ultraScan.

.PARAMETER StartMenu
    Also create the shortcut under the Start menu, so it turns up in search.

.PARAMETER Name
    What to call it. The Microsoft Store build of Friture, if installed, is a
    separate app with its own identity and is not affected either way -- but
    if you have both, give this one a name you can tell apart.

.PARAMETER AppId
    Taskbar identity. MUST match myappid in friture/analyzer.py.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\make_shortcut.ps1 -StartMenu
#>
[CmdletBinding()]
param(
    [switch]$StartMenu,
    [string]$Name = "Friture",
    [string]$AppId = "Friture.Friture.Friture.current"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$script = Join-Path $root "scripts\friture_app.py"
$icon = Join-Path $root "resources\images\friture.ico"

if (-not (Test-Path $script)) { throw "not found: $script" }
if (-not (Test-Path $icon)) { throw "icon not found: $icon" }

# Same search order as friture.bat, and for the same reason: the pythonw on
# PATH is the plain install, which has no PyQt5.
function Find-Pythonw {
    $candidates = @()
    if ($env:FRITURE_PYTHON) {
        $candidates += (Join-Path (Split-Path -Parent $env:FRITURE_PYTHON) "pythonw.exe")
    }
    $candidates += (Join-Path $root ".venv\Scripts\pythonw.exe")
    $candidates += (Join-Path (Split-Path -Parent $root) "ultraScan\.venv\Scripts\pythonw.exe")
    $onPath = Get-Command pythonw.exe -ErrorAction SilentlyContinue
    if ($onPath) { $candidates += $onPath.Source }

    foreach ($candidate in $candidates) {
        if (-not (Test-Path $candidate)) { continue }
        # pythonw writes nothing anywhere we can see it, so ask its python.exe
        $python = Join-Path (Split-Path -Parent $candidate) "python.exe"
        if (-not (Test-Path $python)) { continue }
        & $python -c "import PyQt5" 2>$null
        if ($LASTEXITCODE -eq 0) { return $candidate }
    }
    return $null
}

$pythonw = Find-Pythonw
if (-not $pythonw) {
    throw "No pythonw.exe with PyQt5 found. Set FRITURE_PYTHON, or create a venv here: python -m venv .venv"
}

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

namespace FritureShortcut {

  [StructLayout(LayoutKind.Sequential)]
  public struct PropertyKey { public Guid fmtid; public uint pid; }

  // Only the VT_LPWSTR shape is needed here: a 16-bit tag, padding, a pointer.
  [StructLayout(LayoutKind.Sequential)]
  public struct PropVariant {
    public ushort vt; public ushort r1, r2, r3; public IntPtr p; public IntPtr p2;
  }

  [ComImport, Guid("0000010b-0000-0000-C000-000000000046"),
   InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  public interface IPersistFile {
    void GetClassID(out Guid pClassID);
    [PreserveSig] int IsDirty();
    void Load([MarshalAs(UnmanagedType.LPWStr)] string f, uint mode);
    void Save([MarshalAs(UnmanagedType.LPWStr)] string f, [MarshalAs(UnmanagedType.Bool)] bool remember);
    void SaveCompleted([MarshalAs(UnmanagedType.LPWStr)] string f);
    void GetCurFile([MarshalAs(UnmanagedType.LPWStr)] out string f);
  }

  [ComImport, Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"),
   InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  public interface IPropertyStore {
    void GetCount(out uint c);
    void GetAt(uint i, out PropertyKey key);
    void GetValue(ref PropertyKey key, out PropVariant pv);
    void SetValue(ref PropertyKey key, ref PropVariant pv);
    void Commit();
  }

  // Full vtable order matters even for the methods we never call.
  [ComImport, Guid("000214F9-0000-0000-C000-000000000046"),
   InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  public interface IShellLinkW {
    void GetPath([Out, MarshalAs(UnmanagedType.LPWStr)] System.Text.StringBuilder f,
                 int cch, IntPtr fd, uint flags);
    void GetIDList(out IntPtr ppidl);
    void SetIDList(IntPtr pidl);
    void GetDescription([Out, MarshalAs(UnmanagedType.LPWStr)] System.Text.StringBuilder s, int cch);
    void SetDescription([MarshalAs(UnmanagedType.LPWStr)] string s);
    void GetWorkingDirectory([Out, MarshalAs(UnmanagedType.LPWStr)] System.Text.StringBuilder d, int cch);
    void SetWorkingDirectory([MarshalAs(UnmanagedType.LPWStr)] string d);
    void GetArguments([Out, MarshalAs(UnmanagedType.LPWStr)] System.Text.StringBuilder a, int cch);
    void SetArguments([MarshalAs(UnmanagedType.LPWStr)] string a);
    void GetHotkey(out short k);
    void SetHotkey(short k);
    void GetShowCmd(out int c);
    void SetShowCmd(int c);
    void GetIconLocation([Out, MarshalAs(UnmanagedType.LPWStr)] System.Text.StringBuilder p, int cch, out int i);
    void SetIconLocation([MarshalAs(UnmanagedType.LPWStr)] string p, int i);
    void SetRelativePath([MarshalAs(UnmanagedType.LPWStr)] string p, uint reserved);
    void Resolve(IntPtr hwnd, uint flags);
    void SetPath([MarshalAs(UnmanagedType.LPWStr)] string p);
  }

  [ComImport, Guid("00021401-0000-0000-C000-000000000046")]
  public class ShellLink { }

  public static class Maker {
    // System.AppUserModel.ID -- the taskbar identity a pinned .lnk must carry.
    static readonly Guid APPUSERMODEL =
        new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3");

    public static void Create(string lnk, string target, string args,
                              string workdir, string icon, string desc, string appId) {
      var link = (IShellLinkW)new ShellLink();
      link.SetPath(target);
      link.SetArguments(args);
      link.SetWorkingDirectory(workdir);
      link.SetIconLocation(icon, 0);
      link.SetDescription(desc);

      var store = (IPropertyStore)link;
      var key = new PropertyKey { fmtid = APPUSERMODEL, pid = 5 };
      var pv = new PropVariant { vt = 31 /* VT_LPWSTR */,
                                 p = Marshal.StringToCoTaskMemUni(appId) };
      try {
        store.SetValue(ref key, ref pv);
        store.Commit();
      } finally {
        Marshal.FreeCoTaskMem(pv.p);
      }

      ((IPersistFile)link).Save(lnk, true);
    }
  }
}
'@

$targets = @([pscustomobject]@{
    Where = "デスクトップ"
    Path  = Join-Path ([Environment]::GetFolderPath("Desktop")) ("{0}.lnk" -f $Name)
})
if ($StartMenu) {
    $programs = Join-Path ([Environment]::GetFolderPath("StartMenu")) "Programs"
    $targets += [pscustomobject]@{ Where = "スタートメニュー"; Path = Join-Path $programs ("{0}.lnk" -f $Name) }
}

foreach ($target in $targets) {
    $parent = Split-Path -Parent $target.Path
    if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    [FritureShortcut.Maker]::Create(
        $target.Path, $pythonw, ('"{0}"' -f $script), $root, $icon,
        "Friture リアルタイム音声解析（帯域リスニング対応ビルド）", $AppId)
    Write-Host ("{0,-16} -> {1}" -f $target.Where, $target.Path)
}

Write-Host ""
Write-Host "起動に使う Python : $pythonw"
Write-Host "AppUserModelID    : $AppId  (friture/analyzer.py の myappid と一致していること)"
Write-Host ""
Write-Host "できたショートカットは右クリックから「タスクバーにピン留めする」で常駐できます。"

# Asked of Python rather than spelled out: a Store-based interpreter has its
# %LOCALAPPDATA% redirected into the package's LocalCache, so the obvious path
# is not where the file actually lands.
$python = Join-Path (Split-Path -Parent $pythonw) "python.exe"
$logDir = & $python -c "import platformdirs; print(platformdirs.user_log_dir('Friture', ''))"
Write-Host ("起動しないときのログ: {0}\friture-launch.log" -f $logDir)
