; Uninstall hooks for the NSIS installer.
;
; The app stages coucou-hook.exe into %LOCALAPPDATA%\Coucou\bin at launch, so the
; installer never recorded it and the default uninstaller leaves it behind. The
; inbox and the log live in the same place and are ours too.
;
; Claude Code's own settings.json is deliberately NOT touched here: it belongs to
; the user, it may contain hooks from other tools, and rewriting somebody's
; config from an uninstaller with no diff and no consent is exactly what the rest
; of this app goes out of its way not to do. A relay that is gone exits 0 without
; printing anything, so a leftover entry costs nothing beyond a dead path.

!macro NSIS_HOOK_POSTINSTALL
  ; The default Tauri installer puts a shortcut in the Start Menu but leaves the
  ; Desktop alone unless the install is silent — so a normal double-click install
  ; gave the user nothing to click. This app lives in a corner of the screen and
  ; is started and stopped like any other program, so it gets one either way.
  CreateShortcut "$DESKTOP\Coucou.lnk" "$INSTDIR\${MAINBINARYNAME}.exe"
  !insertmacro SetLnkAppUserModelId "$DESKTOP\Coucou.lnk"
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  RMDir /r "$LOCALAPPDATA\Coucou\bin"
  RMDir /r "$LOCALAPPDATA\Coucou\inbox"
  Delete "$LOCALAPPDATA\Coucou\coucou.log"
!macroend
