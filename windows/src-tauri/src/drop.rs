//! Direct OLE drop target on the island's child windows.
//!
//! wry registers its target once, at webview creation, on whichever child
//! windows exist at that instant. WebView2 creates its leaf windows
//! (`Chrome_RenderWidgetHostHWND` and friends) afterwards, OLE hit-tests to
//! the innermost window and never walks up to the parent, and those leaves
//! carry WebView2's own target — which rejects external drops because wry
//! called `SetAllowExternalDrop(false)`. Result: the "no drop" cursor and no
//! `tauri://drag-*` event ever reaching the webview.
//!
//! So we take the leaf windows ourselves: revoke whoever is there, register
//! ours, and forward the four OLE callbacks as the very events the frontend
//! already listens for. Re-install only when the child window set changes
//! (WebView2 boots async, restarts its renderer), so a call while nothing
//! changed — even mid-drag — is a no-op.

use std::cell::{Cell, RefCell};
use std::ptr;

use serde_json::json;
use tauri::{AppHandle, Emitter, EventTarget, Manager};
use windows::Win32::Foundation::{HWND, LPARAM, POINT, POINTL};
use windows::Win32::Graphics::Gdi::ScreenToClient;
use windows::Win32::System::Com::{IDataObject, DVASPECT_CONTENT, FORMATETC, TYMED_HGLOBAL};
use windows::Win32::System::Ole::{
    CF_HDROP, DROPEFFECT, DROPEFFECT_COPY, DROPEFFECT_NONE, IDropTarget, IDropTarget_Impl,
    RegisterDragDrop, ReleaseStgMedium, RevokeDragDrop,
};
use windows::Win32::System::SystemServices::MODIFIERKEYS_FLAGS;
use windows::Win32::UI::Shell::{DragQueryFileW, HDROP};
use windows::Win32::UI::WindowsAndMessaging::EnumChildWindows;
use windows::core::{BOOL, implement, Result as WinResult};

/// The registered targets have to outlive the registration; OLE holds a ref
/// too, but keeping ours lets a re-install replace them instead of leaking.
struct Owned {
    targets: Vec<IDropTarget>,
    hwnds: Vec<isize>,
}

thread_local! {
    static OWNED: RefCell<Owned> = const {
        RefCell::new(Owned { targets: Vec::new(), hwnds: Vec::new() })
    };
}

/// Register our drop target on every child window of the island, but only
/// when the set of children differs from what we registered last time.
/// Must run on the main (STA) thread — OLE registers per-thread.
pub fn install(app: &AppHandle) {
    let Some(win) = app.get_webview_window(crate::island::WINDOW_LABEL) else {
        return;
    };
    let Some(parent) = crate::island::hwnd_of(&win) else { return };

    let children = child_windows(parent);
    let ids: Vec<isize> = children.iter().map(|h| h.0 as isize).collect();

    let changed = OWNED.with(|o| {
        let o = o.borrow();
        o.hwnds != ids
    });
    if !changed {
        return;
    }

    // Release whoever is registered right now — ours, wry's, WebView2's.
    for &h in &children {
        let _ = unsafe { RevokeDragDrop(h) };
    }
    OWNED.with(|o| o.borrow_mut().targets.clear());

    let mut ok = 0;
    for &h in &children {
        let target: IDropTarget = DragDropTarget {
            hwnd: h,
            app: app.clone(),
            label: crate::island::WINDOW_LABEL.to_string(),
            valid: Cell::new(false),
            effect: Cell::new(DROPEFFECT_NONE),
        }
        .into();
        if unsafe { RegisterDragDrop(h, &target) }.is_ok() {
            OWNED.with(|o| {
                let mut o = o.borrow_mut();
                o.targets.push(target);
            });
            ok += 1;
        }
    }
    OWNED.with(|o| o.borrow_mut().hwnds = ids);
    crate::log::line(format!(
        "drop targets installed on {ok}/{} child windows",
        children.len()
    ));
}

/// All descendants of `parent`, recursively (OLE hit-tests the innermost one).
fn child_windows(parent: HWND) -> Vec<HWND> {
    let mut out: Vec<HWND> = Vec::new();
    {
        let mut closure = |h: HWND| {
            out.push(h);
            true
        };
        let mut trait_obj: &mut dyn FnMut(HWND) -> bool = &mut closure;
        let closure_pointer_pointer: *mut std::ffi::c_void =
            unsafe { std::mem::transmute(&mut trait_obj) };
        unsafe extern "system" fn enumerate_callback(h: HWND, lparam: LPARAM) -> BOOL {
            let f = &mut *(lparam.0 as *mut std::ffi::c_void as *mut &mut dyn FnMut(HWND) -> bool);
            (*f)(h).into()
        }
        let _ = unsafe {
            EnumChildWindows(
                Some(parent),
                Some(enumerate_callback),
                LPARAM(closure_pointer_pointer as isize),
            )
        };
    }
    out
}

/// CF_HDROP paths of a drag data object, or `None` when it carries no files.
fn read_paths(data: &windows::core::Ref<'_, IDataObject>) -> Option<Vec<String>> {
    let format = FORMATETC {
        cfFormat: CF_HDROP.0,
        ptd: ptr::null_mut(),
        dwAspect: DVASPECT_CONTENT.0,
        lindex: -1,
        tymed: TYMED_HGLOBAL.0 as u32,
    };
    let obj = data.as_ref()?;
    let mut medium = unsafe { obj.GetData(&format) }.ok()?;
    let mut paths = Vec::new();
    unsafe {
        let hdrop = HDROP(medium.u.hGlobal.0 as _);
        let count = DragQueryFileW(hdrop, 0xFFFF_FFFF, None);
        for i in 0..count {
            let len = DragQueryFileW(hdrop, i, None) as usize;
            let mut buf = vec![0u16; len + 1];
            DragQueryFileW(hdrop, i, Some(&mut buf));
            paths.push(String::from_utf16_lossy(&buf[..len]));
        }
        let _ = ReleaseStgMedium(&mut medium);
    }
    Some(paths)
}

#[implement(IDropTarget)]
struct DragDropTarget {
    hwnd: HWND,
    app: AppHandle,
    label: String,
    /// Whether the current drag session carries files (mirrors wry's gate).
    valid: Cell<bool>,
    effect: Cell<DROPEFFECT>,
}

impl DragDropTarget {
    fn emit(&self, name: &str, payload: serde_json::Value) {
        let label = self.label.clone();
        let _ = self.app.emit_filter(name, payload, move |t| {
            matches!(
                t,
                EventTarget::Webview { label: l } | EventTarget::WebviewWindow { label: l }
                    if l == &label
            )
        });
    }

    fn to_client(&self, pt: &POINTL) -> (f64, f64) {
        let mut p = POINT { x: pt.x, y: pt.y };
        unsafe {
            let _ = ScreenToClient(self.hwnd, &mut p);
        }
        (p.x as f64, p.y as f64)
    }
}

#[allow(non_snake_case)]
impl IDropTarget_Impl for DragDropTarget_Impl {
    fn DragEnter(
        &self,
        pdataobj: windows::core::Ref<'_, IDataObject>,
        _grfkeystate: MODIFIERKEYS_FLAGS,
        pt: &POINTL,
        pdweffect: *mut DROPEFFECT,
    ) -> WinResult<()> {
        let (x, y) = self.to_client(pt);
        let paths = read_paths(&pdataobj);
        let valid = paths.is_some();
        self.valid.set(valid);
        let effect = if valid { DROPEFFECT_COPY } else { DROPEFFECT_NONE };
        self.effect.set(effect);
        unsafe { *pdweffect = effect };
        if let Some(paths) = paths {
            self.emit(
                "tauri://drag-enter",
                json!({ "paths": paths, "position": { "x": x, "y": y } }),
            );
        }
        Ok(())
    }

    fn DragOver(
        &self,
        _grfkeystate: MODIFIERKEYS_FLAGS,
        pt: &POINTL,
        pdweffect: *mut DROPEFFECT,
    ) -> WinResult<()> {
        let effect = self.effect.get();
        unsafe { *pdweffect = effect };
        if self.valid.get() {
            let (x, y) = self.to_client(pt);
            self.emit("tauri://drag-over", json!({ "position": { "x": x, "y": y } }));
        }
        Ok(())
    }

    fn DragLeave(&self) -> WinResult<()> {
        if self.valid.replace(false) {
            self.effect.set(DROPEFFECT_NONE);
            self.emit("tauri://drag-leave", json!(null));
        }
        Ok(())
    }

    fn Drop(
        &self,
        pdataobj: windows::core::Ref<'_, IDataObject>,
        _grfkeystate: MODIFIERKEYS_FLAGS,
        pt: &POINTL,
        pdweffect: *mut DROPEFFECT,
    ) -> WinResult<()> {
        let effect = self.effect.get();
        unsafe { *pdweffect = effect };
        if self.valid.replace(false) {
            let (x, y) = self.to_client(pt);
            let paths = read_paths(&pdataobj).unwrap_or_default();
            self.emit(
                "tauri://drag-drop",
                json!({ "paths": paths, "position": { "x": x, "y": y } }),
            );
        }
        Ok(())
    }
}
