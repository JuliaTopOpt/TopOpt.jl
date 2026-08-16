# Handle to the figure of the most recently built static-viewer app.
# The figure lives inside the Bonito session handler and is otherwise
# unreachable from Julia; this enables inspection of a live-served app
# (used by tests to verify the JS->Julia camera command channel).
const _last_static_fig = Ref{Any}(nothing)

"""
    _static_visualization(problem; kwargs...)

Build a `Bonito.App` around `visualize(problem; interactive=false, kwargs...)`
with a full camera-control UI implemented in client-side JavaScript, so it
keeps working in statically exported HTML with no running Julia process
(e.g. Quarto/Documenter pages).

Controls mirror the live (Julia-backed) UI:
- Reset, Recenter
- editable camera fields (azimuth φ, elevation θ in degrees; eye x/y/z),
  kept in sync with mouse-driven camera motion
- view presets (Iso / Front / Back / Left / Right / Top / Bottom)
- orthographic toggle (approximated with a 1° telephoto perspective — see
  the implementation note in the JS bridge below)
- "Zoom to cursor" toggle: when ON, wheel/pinch zoom is anchored to the
  pointer position; button zoom always targets the lookat (center).
- Pan/zoom cross to the right of the figure as its own column.
- Save: downloads the WebGL canvas as a PNG with an editable filename

The whole app is centered in the browser viewport (both axes). Requires the
WGLMakie backend (`using WGLMakie`). For a self-contained page, call
`Bonito.Page(exportable=true, offline=true)` first.
"""
function TopOpt._static_visualization(
    problem::AbstractTopOptProblem;
    # Pulled out of kwargs so the in-app legend swatches can reference
    # them — matching the Stiffness visualize defaults. Passing them on
    # to `visualize` keeps the rendered colors the same as without
    # the static viewer.
    undeformed_mesh_color=_default_und_mesh_color(problem),
    load_arrow_color=RGBAf(0.72, 0.12, 0.1, 1.0),
    support_arrow_color=RGBAf(0.72, 0.5, 0.02, 1.0),
    lighting=:none,
    draw_legend=true,
    kw...,
)
    backend = current_backend()
    occursin("WGL", string(backend)) || throw(
        ArgumentError(
            "`visualize(...; static=true)` requires the WGLMakie backend (found $(backend)); load it with `using WGLMakie`",
        ),
    )
    WGLMakie = backend
    Bonito = WGLMakie.Bonito
    D = Bonito.DOM
    # The js"..." macro resolves its module at parse time, which is impossible
    # here (Bonito is only reachable at runtime through the backend module),
    # so the JS is built as raw JSCode with values inlined as JS literals.
    jsvec(v) = "[" * join(Float64.(v), ",") * "]"

    return Bonito.App() do session
        fig = TopOpt.visualize(
            problem;
            interactive=false,
            undeformed_mesh_color=undeformed_mesh_color,
            draw_legend=false,
            load_arrow_color=load_arrow_color,
            support_arrow_color=support_arrow_color,
            lighting=lighting,
            kw...,
        )
        _last_static_fig[] = fig

        # HTML legend overlay rendered in the same coordinate system as the
        # cross, so the two stay aligned as the window resizes. The
        # Makie-internal Legend is disabled above because its grid column
        # width varies with the canvas size and would drift from the overlay.
        css_color(c) = begin
            cc = Makie.to_color(c)
            r = round(Int, 255 * Float64(cc.r))
            g = round(Int, 255 * Float64(cc.g))
            b = round(Int, 255 * Float64(cc.b))
            a = hasproperty(cc, :a) ? Float64(cc.a) : 1.0
            "rgba($r,$g,$b,$a)"
        end
        swatch = "width:12px;height:12px;flex:0 0 auto;border-radius:50%;"
        legend_row = "display:flex;align-items:center;gap:5px;font-size:11px;color:#333;"
        legend = D.div(
            D.div(
                D.div(;
                    style=swatch *
                          "border-radius:0;background:$(css_color(undeformed_mesh_color));",
                ),
                D.span("undeformed mesh"; style="user-select:none;");
                style=legend_row,
            ),
            D.div(
                D.div(; style=swatch * "background:$(css_color(load_arrow_color));"),
                D.span("load arrows"; style="user-select:none;");
                style=legend_row,
            ),
            D.div(
                D.div(; style=swatch * "background:$(css_color(support_arrow_color));"),
                D.span("support arrows"; style="user-select:none;");
                style=legend_row,
            );
            style=join([
                "position:absolute;",
                "top:8px;",
                "right:10px;",
                "z-index:15;",
                "display:flex;",
                "flex-direction:column;",
                "gap:4px;",
                "padding:6px;",
                "background:rgba(255,255,255,0.85);",
                "border:1px solid rgba(0,0,0,0.12);",
                "border-radius:4px;",
                "line-height:1;",
            ]),
        )
        no_legend = D.div()

        ax1_candidates = [c for c in fig.content if c isa LScene]
        # 2D problems use an Axis, not an LScene; there is no 3D camera to
        # control, so return the bare figure with the legend overlay.
        isempty(ax1_candidates) &&
            return D.div(fig, draw_legend ? legend : no_legend; style="position:relative;")
        ax1 = first(ax1_candidates)

        scene_id = WGLMakie.js_uuid(ax1.scene)

        cam = ax1.scene.camera_controls
        lookat = Float64.(cam.lookat[])
        persp_fov = Float64(cam.fov[])

        # Two fixed control rows fit the inline Quarto viewer without wrapping.
        btn = "font-size:10px;padding:1px 3px;margin:0;cursor:pointer;"
        # `num` width fits an optional sign + 3 integer digits + 1 decimal:
        # angles span −180..180 ("−180.0") and positions can reach ±999.9.
        num = "font-size:10px;width:4.5em;padding:1px 2px;"
        lab = "font-size:10px;user-select:none;cursor:pointer;white-space:nowrap;"
        row = "display:flex;flex-wrap:nowrap;gap:3px;align-items:center;justify-content:center;margin:2px 0;"
        cross_btn = "font-size:10px;padding:0;margin:0;cursor:pointer;width:22px;height:20px;"

        # Bounds for the eye-position fields: generous but finite, so typing
        # or spinning cannot fling the camera to numerically absurd places.
        # Reset frames the full mesh bounding box from an isometric view.
        bbox_lo, bbox_hi = boundingbox(problem.ch.dh.grid)
        center = (
            Float64(bbox_hi[1] + bbox_lo[1]) / 2,
            Float64(bbox_hi[2] + bbox_lo[2]) / 2,
            Float64(bbox_hi[3] + bbox_lo[3]) / 2,
        )
        span = (
            Float64(bbox_hi[1] - bbox_lo[1]),
            Float64(bbox_hi[2] - bbox_lo[2]),
            Float64(bbox_hi[3] - bbox_lo[3]),
        )
        # True isometric: elevation atan(1/√2) ≈ 35.264° and azimuth 45°
        # put the eye equally down all three principal axes (the three eye
        # components have the same magnitude). This is the canonical
        # engineering / textbook isometric view.
        el = atand(1 / sqrt(2))
        az = 45.0
        dir_xyz = (cosd(el) * cosd(az), cosd(el) * sind(az), sind(el))
        radius = sqrt(sum(abs2, span)) / 2
        eye_dist = 1.05 * radius / sind(persp_fov / 2)
        lookat = center
        eye0 = (
            lookat[1] + dir_xyz[1] * eye_dist,
            lookat[2] + dir_xyz[2] * eye_dist,
            lookat[3] + dir_xyz[3] * eye_dist,
        )
        cam.lookat[] = Vec3f(lookat...)
        cam.eyeposition[] = Vec3f(eye0...)
        reach = 20 * eye_dist
        function coord_input(class, i)
            return D.input(;
                type="number",
                class,
                style=num,
                min=round(lookat[i] - reach; digits=1),
                max=round(lookat[i] + reach; digits=1),
                step="1",
            )
        end

        button(label, class) = D.button(label; class, style=btn)
        controls = D.div(
            D.div(
                button("Reset", "tv-reset"),
                button("Recenter", "tv-recenter"),
                D.span("Camera"; style=lab * "margin-left:8px;"),
                D.span("φ"; style=lab),
                D.input(;
                    type="number", class="tv-phi", style=num, min=-180, max=180, step="1"
                ),
                D.span("θ"; style=lab),
                D.input(;
                    type="number",
                    class="tv-theta",
                    style=num,
                    min=-89.9,
                    max=89.9,
                    step="1",
                ),
                D.span("x"; style=lab),
                coord_input("tv-x", 1),
                D.span("y"; style=lab),
                coord_input("tv-y", 2),
                D.span("z"; style=lab),
                coord_input("tv-z", 3);
                style=row,
            ),
            D.div(
                (
                    button(p, "tv-preset-$p") for
                    p in ("Iso", "Front", "Back", "Left", "Right", "Top", "Bottom")
                )...,
                D.label(
                    D.input(; type="checkbox", class="tv-ortho"),
                    " Orthographic";
                    style=lab * "margin-left:8px;",
                ),
                D.label(
                    D.input(; type="checkbox", checked=false, class="tv-zoomcursor"),
                    " Zoom to cursor";
                    style=lab,
                ),
                D.input(;
                    type="text",
                    value="topopt_view.png",
                    class="tv-savename",
                    style="font-size:10px;width:10em;padding:1px 2px;margin-left:5px;",
                ),
                button("Save", "tv-save");
                style=row,
            );
            style="position:relative;z-index:100;width:580px;flex:0 0 auto;overflow:visible;margin-top:6px;",
        )

        # Overlay the controls so the 3D viewport uses the full inline width.
        cross = D.div(
            D.div(
                D.button("−"; class="tv-zoomout", style=cross_btn),
                D.button("+"; class="tv-zoomin", style=cross_btn);
                style="display:flex;gap:6px;justify-content:center;margin-bottom:4px;",
            ),
            D.div(
                D.button("↑"; class="tv-panup", style=cross_btn);
                style="display:flex;justify-content:center;margin-top:5px;margin-bottom:5px;",
            ),
            D.div(
                D.button("←"; class="tv-panleft", style=cross_btn),
                D.button("→"; class="tv-panright", style=cross_btn);
                style="display:flex;gap:12px;justify-content:center;",
            ),
            D.div(
                D.button("↓"; class="tv-pandown", style=cross_btn);
                style="display:flex;justify-content:center;margin-top:5px;",
            );
            style=join([
                "position:absolute;",
                "right:45px;",
                "bottom:0;",
                "z-index:20;",
                "width:60px;",
                "padding:4px 0;",
                "display:flex;",
                "flex-direction:column;",
                "align-items:center;",
            ]),
        )

        # The legend and controls share the right side of the static viewport.
        # Both anchor to the figure so they stay aligned as the window resizes.
        figure = D.div(
            fig,
            draw_legend ? legend : no_legend,
            cross;
            style=join([
                "position:relative;",
                "flex:0 0 auto;",
                "min-width:0;",
                "min-height:0;",
                "width:100%;",
                "max-width:1170px;",
                "margin:0 auto;",
                "display:flex;",
                "justify-content:center;",
                "line-height:0;",
            ]),
        )
        viewport = D.div(
            figure;
            style=join([
                "display:flex;",
                "align-items:flex-end;",
                "justify-content:center;",
                "width:100%;",
                "min-width:0;",
                "min-height:0;",
                "flex:0 0 auto;",
                "gap:6px;",
                "margin:0;",
            ]),
        )
        container = D.div(
            controls,
            viewport;
            style=join([
                "position:relative;",
                "display:flex;",
                "flex-direction:column;",
                "justify-content:flex-start;",
                "align-items:center;",
                "gap:8px;",
                "padding:8px;",
                "box-sizing:border-box;",
                "width:100%;",
                "max-width:1170px;",
                "margin:0 auto;",
                "overflow:visible;",
            ]),
        )

        # --- live/offline camera bridge -----------------------------------
        # In a static export the JS THREE.OrbitControls is the camera
        # authority. In a *live* session (VSCode/Quarto inline, browser with
        # a running Julia process) WGLMakie disables the JS OrbitControls
        # (its update() no-ops while `Bonito.can_send_to_julia()`), and
        # Julia's Camera3D is the authority instead. Every control therefore
        # branches: offline → drive OrbitControls; live → send a command
        # through `cam_cmd`, applied here via `update_cam!`. `cam_state`
        # mirrors the Julia camera back to JS so the readout fields track
        # mouse motion and so JS computes new poses from fresh state.
        #
        # cam_cmd:  [1, eye..., target..., up...]  set pose (up=0,0,0 keeps up)
        #           [2, flag]                      orthographic on/off
        # cam_state: [eye..., target..., up..., fov, ortho_flag]
        function state_vec()
            return Float64[
                cam.eyeposition[]...,
                cam.lookat[]...,
                cam.upvector[]...,
                cam.fov[],
                cam.settings.projectiontype[] == Makie.Orthographic ? 1.0 : 0.0,
            ]
        end
        cam_cmd = Bonito.Observable(Float64[])
        cam_state = Bonito.Observable(state_vec())
        sync_state!(_...) = (cam_state[]=state_vec(); nothing)
        on(cam_cmd) do v
            isempty(v) && return nothing
            op = Int(v[1])
            if op == 1
                length(v) == 10 || throw(
                    ArgumentError("camera pose command needs 10 values, got $(length(v))"),
                )
                eye = Vec3f(v[2], v[3], v[4])
                tgt = Vec3f(v[5], v[6], v[7])
                up = Vec3f(v[8], v[9], v[10])
                if norm(up) < 0.5
                    update_cam!(ax1.scene, cam, eye, tgt)
                else
                    update_cam!(ax1.scene, cam, eye, tgt, up)
                end
            elseif op == 2
                checked = v[2] > 0.5
                is_ortho = cam.settings.projectiontype[] == Makie.Orthographic
                if checked != is_ortho
                    # Same distance rescale as the live UI toggle: ortho's
                    # visible half-height at distance d is d itself,
                    # perspective's is d*tand(fov/2).
                    fov_scale = tand(0.5 * cam.fov[])
                    eyev = Vec3f(cam.eyeposition[])
                    lav = Vec3f(cam.lookat[])
                    dirv = eyev - lav
                    d = norm(dirv)
                    nd = checked ? d * fov_scale : d / fov_scale
                    cam.eyeposition[] = lav + (dirv / d) * nd
                    cam.settings.projectiontype[] =
                        checked ? Makie.Orthographic : Makie.Perspective
                end
            else
                throw(ArgumentError("unknown camera command opcode $op"))
            end
            sync_state!()
            return nothing
        end
        # Mirror mouse-driven camera motion (throttled: each sync rewrites
        # five textboxes in JS).
        on(sync_state!, throttle(0.3, cam.eyeposition))
        on(sync_state!, throttle(0.3, cam.lookat))

        # All behavior is wired in one onload script: query controls by
        # class, poll until WGLMakie has registered the scene (the WebGL
        # canvas initializes asynchronously), then attach listeners.
        setup_js = """
            const q = (cls) => container.querySelector('.' + cls);
            const perspFov = $(persp_fov);
            const la0 = $(jsvec(lookat));
            const eye0 = $(jsvec(eye0));
            const deg = Math.PI / 180;
            // Offline orthographic approximation: WGLMakie's JS camera sync
            // is a closure over a PerspectiveCamera, so a true
            // OrthographicCamera cannot be swapped in; a 1° telephoto is
            // visually indistinguishable from parallel projection. Live
            // sessions use Makie's real Orthographic instead (op 2).
            const ORTHO_FOV = 1.0;
            const live = () => Bonito.can_send_to_julia && Bonito.can_send_to_julia();

            const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
            const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
            const scl = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
            const len = (a) => Math.hypot(a[0], a[1], a[2]);
            const cross = (a, b) => [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ];
            const normv = (a) => scl(a, 1 / len(a));

            const poll = setInterval(() => {
                const scene = window.WGL.find_scene('$(scene_id)');
                if (!scene || !scene.orbitcontrols) return;
                clearInterval(poll);
                const c = scene.orbitcontrols;

                // Current camera state, from Julia in live sessions (the JS
                // OrbitControls is stale there) or from OrbitControls offline.
                const cur = () => {
                    if (live()) {
                        const s = camState.value;
                        return {
                            eye: s.slice(0, 3), tgt: s.slice(3, 6), up: s.slice(6, 9),
                            fov: s[9], ortho: s[10] > 0.5,
                        };
                    }
                    const p = c.object.position, t = c.target, u = c.object.up;
                    return {
                        eye: [p.x, p.y, p.z], tgt: [t.x, t.y, t.z], up: [u.x, u.y, u.z],
                        fov: c.object.fov, ortho: c.object.fov < perspFov - 1e-3,
                    };
                };
                // Visible half-height at the lookat plane; pan steps scale
                // with it so one click moves ~10% of the view in either mode.
                const halfH = () => {
                    const s = cur();
                    const d = len(sub(s.eye, s.tgt));
                    return (live() && s.ortho) ? d : d * Math.tan(s.fov * deg / 2);
                };
                const setPose = (eye, tgt, up) => {  // up = null keeps it
                    if (live()) {
                        camCmd.notify([1, ...eye, ...tgt, ...(up || [0, 0, 0])]);
                        return;
                    }
                    if (up) c.object.up.set(up[0], up[1], up[2]);
                    c.object.position.set(eye[0], eye[1], eye[2]);
                    c.target.set(tgt[0], tgt[1], tgt[2]);
                    c.update();
                };

                // --- camera fields ---------------------------------------
                const fields = ['phi', 'theta', 'x', 'y', 'z'].map(n => q('tv-' + n));
                // Sync Julia's camera state to BOTH OrbitControls' actual
                // position/target AND the readout boxes. Without driving
                // `c.object.position` and `c.target` here, OrbitControls'
                // initial pose (whatever Three.js baked in at load) would
                // override whatever Julia set before exporting, and the
                // first visualised frame wouldn't be the canonical view
                // the user asked for.
                const driveCamera = (s, withUp) => {
                    c.object.position.set(s.eye[0], s.eye[1], s.eye[2]);
                    c.target.set(s.tgt[0], s.tgt[1], s.tgt[2]);
                    if (withUp) c.object.up.set(s.up[0], s.up[1], s.up[2]);
                    c.update();
                };
                const syncFields = () => {
                    const s = cur();
                    const d = sub(s.eye, s.tgt), r = len(d);
                    const phi = Math.atan2(d[1], d[0]) / deg;
                    const th = Math.asin(Math.max(-1, Math.min(1, d[2] / r))) / deg;
                    const vals = [phi, th, s.eye[0], s.eye[1], s.eye[2]];
                    fields.forEach((f, i) => {
                        if (document.activeElement !== f) f.value = vals[i].toFixed(1);
                    });
                    q('tv-ortho').checked = s.ortho;
                };
                let pending = false;
                const queueSync = () => {
                    if (pending) return;
                    pending = true;
                    setTimeout(() => { pending = false; syncFields(); }, 150);
                };
                c.addEventListener('change', queueSync);  // offline mouse motion
                camState.on(queueSync);                   // live mouse motion
                // Force OrbitControls' first frame to Julia's canonical
                // eye0/la0 (the Reset target). Without this, the exported
                // Three.js scene would draw whatever baked pose its first
                // load cycle gave it, which is usually a near-front-on
                // view of the bounding-box center rather than the
                // isometric point Julia positioned for. After this snap, any
                // user-driven or `Reset`-driven change keeps the camera
                // and the readout boxes consistent.
                driveCamera(camState.value, true);
                syncFields();

                // Typed values bypass the min/max attributes; clamp here.
                const clampToInput = (v, el) => Math.min(
                    parseFloat(el.max), Math.max(parseFloat(el.min), v));
                const setAngles = (phi, th) => {
                    const s = cur();
                    const r = len(sub(s.eye, s.tgt));
                    th = Math.max(-89.9, Math.min(89.9, th));
                    const ct = Math.cos(th * deg);
                    setPose([
                        s.tgt[0] + r * ct * Math.cos(phi * deg),
                        s.tgt[1] + r * ct * Math.sin(phi * deg),
                        s.tgt[2] + r * Math.sin(th * deg),
                    ], s.tgt, null);
                };
                q('tv-phi').addEventListener('change', (e) => {
                    setAngles(clampToInput(parseFloat(e.target.value), e.target),
                              parseFloat(q('tv-theta').value));
                });
                q('tv-theta').addEventListener('change', (e) => {
                    setAngles(parseFloat(q('tv-phi').value),
                              clampToInput(parseFloat(e.target.value), e.target));
                });
                ['x', 'y', 'z'].forEach((n, i) => {
                    q('tv-' + n).addEventListener('change', (e) => {
                        const s = cur();
                        const eye = s.eye.slice();
                        eye[i] = clampToInput(parseFloat(e.target.value), e.target);
                        setPose(eye, s.tgt, null);
                    });
                });

                // --- orthographic toggle ----------------------------------
                const setOrtho = (flag) => {
                    if (live()) { camCmd.notify([2, flag ? 1 : 0]); return; }
                    const fov = flag ? ORTHO_FOV : perspFov;
                    const oldTan = Math.tan(c.object.fov * deg / 2);
                    const newTan = Math.tan(fov * deg / 2);
                    c.object.fov = fov;
                    c.object.position.lerpVectors(
                        c.target, c.object.position, oldTan / newTan);
                    c.object.updateProjectionMatrix();
                    c.update();
                };
                q('tv-ortho').addEventListener('change', (e) => setOrtho(e.target.checked));

                // --- zoom-to-cursor toggle --------------------------------
                // OrbitControls.zoomToCursor=true anchors wheel/pinch zoom to
                // the pointer; buttons always target the lookat and ignore
                // this flag (setPose() reads from cur().tgt directly).
                const applyZoomCursor = () => {
                    c.zoomToCursor = q('tv-zoomcursor').checked;
                };
                applyZoomCursor();
                q('tv-zoomcursor').addEventListener('change', applyZoomCursor);

                // --- reset / recenter -------------------------------------
                q('tv-reset').addEventListener('click', () => {
                    q('tv-ortho').checked = false;
                    if (live()) {
                        camCmd.notify([2, 0]);
                        camCmd.notify([1, ...eye0, ...la0, 0, 0, 1]);
                        return;
                    }
                    // reset() restores position/target/zoom but not fov; the
                    // saved position only frames correctly at the original fov.
                    c.object.fov = perspFov;
                    c.object.updateProjectionMatrix();
                    c.reset();
                });
                q('tv-recenter').addEventListener('click', () => {
                    const s = cur();
                    const offset = sub(la0, s.tgt);
                    setPose(add(s.eye, offset), la0, null);
                });

                // --- view presets (keep current distance) -----------------
                const isq = 1 / Math.sqrt(3);
                const presets = {
                    Iso: [[isq, isq, isq], [0, 0, 1]],
                    Front: [[0, -1, 0], [0, 0, 1]],
                    Back: [[0, 1, 0], [0, 0, 1]],
                    Left: [[-1, 0, 0], [0, 0, 1]],
                    Right: [[1, 0, 0], [0, 0, 1]],
                    Top: [[0, 0, 1], [0, 1, 0]],
                    Bottom: [[0, 0, -1], [0, 1, 0]],
                };
                for (const [name, [dir, up]] of Object.entries(presets)) {
                    q('tv-preset-' + name).addEventListener('click', () => {
                        const s = cur();
                        const d = len(sub(s.eye, s.tgt));
                        setPose(add(s.tgt, scl(dir, d)), s.tgt, up);
                    });
                }

                // --- zoom / pan cross -------------------------------------
                const zoom = (f) => {
                    const s = cur();
                    setPose(add(s.tgt, scl(sub(s.eye, s.tgt), f)), s.tgt, null);
                };
                q('tv-zoomin').addEventListener('click', () => zoom(0.9));
                q('tv-zoomout').addEventListener('click', () => zoom(1.125));
                const pan = (dx, dy) => {
                    const s = cur();
                    const view = sub(s.tgt, s.eye);
                    const right = normv(cross(view, s.up));
                    const up2 = normv(cross(right, view));
                    const st = halfH() * 0.1;
                    const off = add(scl(right, dx * st), scl(up2, dy * st));
                    setPose(add(s.eye, off), add(s.tgt, off), null);
                };
                q('tv-panleft').addEventListener('click', () => pan(-1, 0));
                q('tv-panright').addEventListener('click', () => pan(1, 0));
                q('tv-panup').addEventListener('click', () => pan(0, 1));
                q('tv-pandown').addEventListener('click', () => pan(0, -1));

                // --- save -------------------------------------------------
                // The WebGL canvas is created with preserveDrawingBuffer, so
                // toDataURL captures the last rendered frame (figure +
                // legend; the HTML controls are outside the canvas).
                q('tv-save').addEventListener('click', () => {
                    const canvas = scene.screen.canvas ||
                        scene.screen.renderer.domElement;
                    const a = document.createElement('a');
                    a.download = q('tv-savename').value || 'topopt_view.png';
                    a.href = canvas.toDataURL('image/png');
                    a.click();
                });
            }, 200);
        """
        # The observables must be interpolated as session objects (not
        # literals) so JS `.notify`/`.on`/`.value` connect to Julia; build
        # the JSCode source vector by hand since js"..." is unavailable here.
        # For static export we want a self-contained script that doesn't rely on
        # `Bonito.init_session` registering `cam_cmd`/`cam_state` (those
        # `__lookup_interpolated` keys fail without the live session). All
        # controls operate directly on `c` (Three.js OrbitControls) in
        # offline mode; the live-session push is best-effort.
        jsvec_inline(v) = "[" * join(Float64.(v), ",") * "]"
        eye0_js = jsvec_inline(eye0)
        la0_js = jsvec_inline(lookat)
        fov_js = string(persp_fov)
        scene_id_js = string(scene_id)
        # Legend entries (label, css color, swatch shape) for compositing onto
        # the saved PNG. The zoom/pan controls are deliberately excluded.
        legend_entries = [
            ("undeformed mesh", css_color(undeformed_mesh_color), "square"),
            ("load arrows", css_color(load_arrow_color), "circle"),
            ("support arrows", css_color(support_arrow_color), "circle"),
        ]
        legend_js =
            "[" *
            join(
                (
                    "{label:\"$(e[1])\",color:\"$(e[2])\",shape:\"$(e[3])\"}" for
                    e in legend_entries
                ),
                ",",
            ) *
            "]"

        container = D.div(
            container,
            D.script(
                """
                function initialize_static_view(container) {
                    const eye0 = $(eye0_js);
                    const la0 = $(la0_js);
                    const fov0 = $(fov_js);
                    const scene_id = $(string("\"", scene_id_js, "\""));
                    const legend_entries = $(legend_js);

                    const ORTHO_FOV = 1.0;
                    const deg = Math.PI / 180;
                    function live() {
                        return typeof Bonito !== 'undefined' &&
                               Bonito.can_send_to_julia &&
                               Bonito.can_send_to_julia();
                    }

                    const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
                    const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
                    const scl = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
                    const len = (a) => Math.hypot(a[0], a[1], a[2]);
                    const cross = (a, b) => [
                        a[1] * b[2] - a[2] * b[1],
                        a[2] * b[0] - a[0] * b[2],
                        a[0] * b[1] - a[1] * b[0],
                    ];
                    const normv = (a) => scl(a, 1 / len(a));

                    const poll = setInterval(() => {
                        if (!window.WGL || typeof window.WGL.find_scene !== 'function') return;
                        const scene = window.WGL.find_scene(scene_id);
                        if (!scene || !scene.orbitcontrols) return;
                        clearInterval(poll);
                        const c = scene.orbitcontrols;

                        // This app owns its camera in the browser. WGLMakie's
                        // live-session guard otherwise disables OrbitControls
                        // and can restore Julia's initial pose after rotation.
                        if (typeof Bonito !== 'undefined' && Bonito.can_send_to_julia) {
                            Bonito.can_send_to_julia = () => false;
                        }

                        const refreshCamera = (notify = true) => {
                            c.object.lookAt(c.target);
                            c.object.updateMatrixWorld();
                            c.update();
                            // WGLMakie disables OrbitControls updates while a
                            // Julia session is connected. Dispatching the
                            // change event still refreshes its camera matrices.
                            if (live() && notify) c.dispatchEvent({type: 'change'});
                        };

                        // Initial-frame driver: take the canonical eye0/lookat
                        // and force OrbitControls onto them so the first paint
                        // matches the Reset target. Without this Three.js keeps
                        // whatever default its first-load cycle produced.
                        const driveCamera = (eye, tgt, up) => {
                            c.object.position.set(eye[0], eye[1], eye[2]);
                            c.target.set(tgt[0], tgt[1], tgt[2]);
                            if (up) c.object.up.set(up[0], up[1], up[2]);
                            refreshCamera();
                        };

                        // setPose: drives the camera directly via OrbitControls.
                        // In offline (static) mode this is the only path and it
                        // works without any Julia session. In live mode the
                        // same OrbitControls write happens — `camera_controls`
                        // is the visual authority either way.
                        const setPose = (eye, tgt, up) => {
                            if (up) c.object.up.set(up[0], up[1], up[2]);
                            c.object.position.set(eye[0], eye[1], eye[2]);
                            c.target.set(tgt[0], tgt[1], tgt[2]);
                            refreshCamera();
                        };

                        const cur = () => {
                            const p = c.object.position, t = c.target, u = c.object.up;
                            return {
                                eye: [p.x, p.y, p.z], tgt: [t.x, t.y, t.z], up: [u.x, u.y, u.z],
                                fov: c.object.fov,
                                ortho: c.object.fov < fov0 - 1e-3,
                            };
                        };

                        const halfH = () => {
                            const s = cur();
                            const d = len(sub(s.eye, s.tgt));
                            return (live() && s.ortho) ? d : d * Math.tan(s.fov * deg / 2);
                        };

                        const fields = ['phi', 'theta', 'x', 'y', 'z'].map(n => container.querySelector('.tv-' + n));
                        const syncFields = () => {
                            const s = cur();
                            const d = sub(s.eye, s.tgt), r = len(d);
                            const phi = Math.atan2(d[1], d[0]) / deg;
                            const th = Math.asin(Math.max(-1, Math.min(1, d[2] / r))) / deg;
                            const vals = [phi, th, s.eye[0], s.eye[1], s.eye[2]];
                            fields.forEach((f, i) => {
                                if (document.activeElement !== f) f.value = vals[i].toFixed(1);
                            });
                            const ortho_el = container.querySelector('.tv-ortho');
                            if (ortho_el) ortho_el.checked = s.ortho;
                        };
                        let pending = false;
                        const queueSync = () => {
                            if (pending) return;
                            pending = true;
                            setTimeout(() => { pending = false; syncFields(); }, 150);
                        };
                        c.addEventListener('change', queueSync);
                        syncFields();

                        const clampToInput = (v, el) =>
                            Math.min(parseFloat(el.max), Math.max(parseFloat(el.min), v));
                        const setAngles = (phi, th) => {
                            const s = cur();
                            const r = len(sub(s.eye, s.tgt));
                            th = Math.max(-89.9, Math.min(89.9, th));
                            const ct = Math.cos(th * deg);
                            setPose([
                                s.tgt[0] + r * ct * Math.cos(phi * deg),
                                s.tgt[1] + r * ct * Math.sin(phi * deg),
                                s.tgt[2] + r * Math.sin(th * deg),
                            ], s.tgt, null);
                        };
                        const fldOrNull = (sel) => {
                            const el = container.querySelector(sel);
                            return el ? el : null;
                        };
                        const phi_el = fldOrNull('.tv-phi');
                        if (phi_el) phi_el.addEventListener('change', (e) => {
                            const theta_el = fldOrNull('.tv-theta');
                            const tv = (theta_el && parseFloat(theta_el.value)) || 30;
                            setAngles(clampToInput(parseFloat(e.target.value), e.target), tv);
                        });
                        const theta_el = fldOrNull('.tv-theta');
                        if (theta_el) theta_el.addEventListener('change', (e) => {
                            const phi_el2 = fldOrNull('.tv-phi');
                            const pv = (phi_el2 && parseFloat(phi_el2.value)) || 45;
                            setAngles(pv, clampToInput(parseFloat(e.target.value), e.target));
                        });
                        ['x', 'y', 'z'].forEach((n, i) => {
                            const el = fldOrNull('.tv-' + n);
                            if (!el) return;
                            const applyCoordinate = () => {
                                const value = parseFloat(el.value);
                                if (!Number.isFinite(value)) return;
                                const s = cur();
                                const eye = s.eye.slice();
                                eye[i] = clampToInput(value, el);
                                el.value = eye[i].toFixed(1);
                                setPose(eye, s.tgt, null);
                            };
                            el.addEventListener('change', applyCoordinate);
                            el.addEventListener('keydown', (e) => {
                                if (e.key === 'Enter') {
                                    e.preventDefault();
                                    applyCoordinate();
                                    el.blur();
                                }
                            });
                        });

                        const setOrtho = (flag) => {
                            const fov = flag ? ORTHO_FOV : fov0;
                            const oldTan = Math.tan(c.object.fov * deg / 2);
                            const newTan = Math.tan(fov * deg / 2);
                            c.object.fov = fov;
                            c.object.position.lerpVectors(c.target, c.object.position, oldTan / newTan);
                            c.object.updateProjectionMatrix();
                            refreshCamera();
                        };
                        const ortho_el = fldOrNull('.tv-ortho');
                        if (ortho_el) ortho_el.addEventListener('change', (e) => setOrtho(e.target.checked));

                        const zoomcursor_el = fldOrNull('.tv-zoomcursor');
                        if (zoomcursor_el) {
                            const applyZoomCursor = () => {
                                c.zoomToCursor = zoomcursor_el.checked;
                            };
                            applyZoomCursor();
                            zoomcursor_el.addEventListener('change', applyZoomCursor);
                        }

                        const reset_el = fldOrNull('.tv-reset');
                        if (reset_el) reset_el.addEventListener('click', () => {
                            const ortho_el2 = fldOrNull('.tv-ortho');
                            if (ortho_el2) ortho_el2.checked = false;
                            c.object.fov = fov0;
                            c.object.updateProjectionMatrix();
                            setPose(eye0, la0, [0, 0, 1]);
                        });
                        const recenter_el = fldOrNull('.tv-recenter');
                        if (recenter_el) recenter_el.addEventListener('click', () => {
                            const s = cur();
                            const offset = sub(la0, s.tgt);
                            setPose(add(s.eye, offset), la0, null);
                        });

                        const isq = 1 / Math.sqrt(3);
                        const presets = {
                            Iso: [[isq, isq, isq], [0, 0, 1]],
                            Front: [[0, -1, 0], [0, 0, 1]],
                            Back: [[0, 1, 0], [0, 0, 1]],
                            Left: [[-1, 0, 0], [0, 0, 1]],
                            Right: [[1, 0, 0], [0, 0, 1]],
                            Top: [[0, 0, 1], [0, 1, 0]],
                            Bottom: [[0, 0, -1], [0, 1, 0]],
                        };
                        Object.entries(presets).forEach(([name, [dir, up]]) => {
                            const btn = fldOrNull('.tv-preset-' + name);
                            if (!btn) return;
                            btn.addEventListener('click', () => {
                                const s = cur();
                                const d = len(sub(s.eye, s.tgt));
                                setPose(add(s.tgt, scl(dir, d)), s.tgt, up);
                            });
                        });

                        const zoom = (f) => {
                            const s = cur();
                            setPose(add(s.tgt, scl(sub(s.eye, s.tgt), f)), s.tgt, null);
                        };
                        const zoomin_el = fldOrNull('.tv-zoomin');
                        if (zoomin_el) zoomin_el.addEventListener('click', () => zoom(0.9));
                        const zoomout_el = fldOrNull('.tv-zoomout');
                        if (zoomout_el) zoomout_el.addEventListener('click', () => zoom(1.125));

                        const pan = (dx, dy) => {
                            const s = cur();
                            const view = sub(s.tgt, s.eye);
                            const right = normv(cross(view, s.up));
                            const up2 = normv(cross(right, view));
                            const st = halfH() * 0.1;
                            const off = add(scl(right, dx * st), scl(up2, dy * st));
                            setPose(add(s.eye, off), add(s.tgt, off), null);
                        };
                        const pan_btns = [
                            ['panleft', -1, 0],
                            ['panright', 1, 0],
                            ['panup', 0, 1],
                            ['pandown', 0, -1],
                        ];
                        pan_btns.forEach(([cls, dx, dy]) => {
                            const btn = fldOrNull('.tv-' + cls);
                            if (!btn) return;
                            btn.addEventListener('click', () => pan(dx, dy));
                        });

                        const save_el = fldOrNull('.tv-save');
                        if (save_el) save_el.addEventListener('click', () => {
                            const canvas = scene.screen.canvas || scene.screen.renderer.domElement;
                            const save_box = fldOrNull('.tv-savename');
                            // Composite the legend (but not the zoom/pan
                            // controls) onto the WebGL frame before export.
                            const out = document.createElement('canvas');
                            out.width = canvas.width;
                            out.height = canvas.height;
                            const ctx = out.getContext('2d');
                            ctx.drawImage(canvas, 0, 0);
                            if (legend_entries.length) {
                                const pad = 6, gap = 4, sw = 12, sh = 12, gapX = 5, radius = 4;
                                ctx.font = '11px sans-serif';
                                const textW = legend_entries.map(e => ctx.measureText(e.label).width);
                                const rowW = legend_entries.map((e, i) => sw + gapX + textW[i]);
                                const lineH = Math.max(sh, 14);
                                const boxW = Math.max(...rowW) + pad * 2;
                                const boxH = legend_entries.length * lineH + (legend_entries.length - 1) * gap + pad * 2;
                                const x0 = canvas.width - boxW - 10;
                                const y0 = 8;
                                ctx.beginPath();
                                ctx.moveTo(x0 + radius, y0);
                                ctx.arcTo(x0 + boxW, y0, x0 + boxW, y0 + boxH, radius);
                                ctx.arcTo(x0 + boxW, y0 + boxH, x0, y0 + boxH, radius);
                                ctx.arcTo(x0, y0 + boxH, x0, y0, radius);
                                ctx.arcTo(x0, y0, x0 + boxW, y0, radius);
                                ctx.closePath();
                                ctx.fillStyle = 'rgba(255,255,255,0.85)';
                                ctx.fill();
                                ctx.strokeStyle = 'rgba(0,0,0,0.12)';
                                ctx.lineWidth = 1;
                                ctx.stroke();
                                legend_entries.forEach((e, i) => {
                                    const cy = y0 + pad + i * (lineH + gap) + lineH / 2;
                                    const cx = x0 + pad + sw / 2;
                                    ctx.fillStyle = e.color;
                                    if (e.shape === 'circle') {
                                        ctx.beginPath();
                                        ctx.arc(cx, cy, sw / 2, 0, 2 * Math.PI);
                                        ctx.fill();
                                    } else {
                                        ctx.fillRect(x0 + pad, cy - sh / 2, sw, sh);
                                    }
                                    ctx.fillStyle = '#333';
                                    ctx.textBaseline = 'middle';
                                    ctx.fillText(e.label, x0 + pad + sw + gapX, cy);
                                });
                            }
                            const a = document.createElement('a');
                            a.download = (save_box && save_box.value) || 'topopt_view.png';
                            a.href = out.toDataURL('image/png');
                            a.click();
                        });

                        // Final: drive OrbitControls into Julia's eye0/lookat
                        // so the very first paint matches the Reset target.
                        driveCamera(eye0, la0, [0, 0, 1]);
                    }, 200);
                }
                // Module scripts do not expose document.currentScript. Find
                // this script by its unique function marker and use its
                // parent as the control/figure container.
                const self = [...document.scripts].find(s =>
                    s.textContent.includes('initialize_static_view'));
                initialize_static_view(self ? self.parentElement : document.body);
                """;
                type="module",
            ),
        )

        return container
    end
end
