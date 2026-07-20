"""
ROBOKINE-INDIA V1.1
Universal Robot Kinematics GUI for limited 6-DOF serial robots with mixed joint support.
Developed by Dr. Priyam Parikh and Aahanan Basappa.

Install: pip install numpy matplotlib pillow
Optional MP4 export: install FFmpeg and add it to PATH.
Run: python ROBOKINE_INDIA_V1_1_Mixed_Joints.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter


# ----------------------------- Math utilities -----------------------------

def rot_x(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def tf(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array(p, dtype=float)
    return T


class RoboKineIndia:
    def __init__(self, root):
        self.root = root
        self.root.title("ROBOKINE-INDIA V1.1 - Mixed Joint Support")
        self.root.geometry("1880x980")

        self.dof = tk.IntVar(value=4)
        self.axis_options = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
        self.joint_type_options = ["Revolute", "Prismatic", "Fixed"]
        self.joint_type_vars, self.slider_widgets = [], []
        self.value_label_widgets = []
        self.axis_vars, self.link_vars, self.home_vars = [], [], []
        self.min_vars, self.max_vars, self.angle_vars = [], [], []

        self.target_x = tk.DoubleVar(value=180.0)
        self.target_y = tk.DoubleVar(value=60.0)
        self.target_z = tk.DoubleVar(value=120.0)
        self.ik_tol = tk.DoubleVar(value=1.0)
        self.ik_iter = tk.IntVar(value=1000)
        self.ik_damp = tk.DoubleVar(value=0.15)

        self.via_x = tk.DoubleVar(value=180.0)
        self.via_y = tk.DoubleVar(value=60.0)
        self.via_z = tk.DoubleVar(value=120.0)
        self.via_points = []
        self.via_q = None
        self.selected_via = None

        self.ws_samples = tk.IntVar(value=3000)
        self.workspace = None
        self.workspace_q = None
        self.workspace_err = None

        self.rank_tol = tk.DoubleVar(value=1e-5)
        self.manip_tol = tk.DoubleVar(value=1e-6)
        self.cond_limit = tk.DoubleVar(value=1000.0)

        self.method = tk.StringVar(value="Fifth Order Polynomial")
        self.segment_time = tk.DoubleVar(value=2.0)
        self.samples_per_segment = tk.IntVar(value=80)
        self.anim_delay = tk.IntVar(value=12)

        self.traj_t = None
        self.traj_q = None
        self.traj_qd = None
        self.traj_qdd = None
        self.traj_seg = None
        self.ee_path = None
        self.frame = 0
        self.running = False
        self.after_id = None

        self.show_global_axes = tk.BooleanVar(value=True)
        self.show_frames = tk.BooleanVar(value=True)
        self.show_rot_axes = tk.BooleanVar(value=True)
        self.show_dims = tk.BooleanVar(value=True)
        self.show_workspace = tk.BooleanVar(value=False)
        self.show_via = tk.BooleanVar(value=True)
        self.show_path = tk.BooleanVar(value=True)
        self.show_home = tk.BooleanVar(value=True)

        self.colors = {
            "x": "#8B0000", "y": "#006400", "z": "#00008B", "robot": "#222222",
            "joint": "#FF8C00", "ee": "#B00020", "workspace": "#777777", "via": "#4B0082",
            "path": "#0047AB", "home": "#999999"
        }

        self.build_gui()
        self.rebuild_joints()
        self.update_scene()
        self.update_realtime_graphs(None)

    # ----------------------------- GUI helpers -----------------------------
    def scroll_panel(self, parent, width):
        outer = ttk.Frame(parent, padding=4)
        canvas = tk.Canvas(outer, width=width, highlightthickness=0)
        bar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)
        bar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner = ttk.Frame(canvas, padding=8)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")

        def configure(_=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(win, width=canvas.winfo_width())
        inner.bind("<Configure>", configure)
        canvas.bind("<Configure>", configure)

        def wheel(event):
            if canvas.winfo_containing(event.x_root, event.y_root) is not None:
                canvas.yview_scroll(int(-event.delta / 120), "units")
        canvas.bind_all("<MouseWheel>", wheel)
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))
        return outer, inner

    def entry_row(self, parent, label, var, command=None):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=30).pack(side=tk.LEFT)
        ent = ttk.Entry(row, textvariable=var, width=12)
        ent.pack(side=tk.RIGHT)
        cmd = command or self.update_scene
        ent.bind("<Return>", lambda e: cmd())
        ent.bind("<FocusOut>", lambda e: cmd())
        return ent

    def build_gui(self):
        self.left_outer, self.left = self.scroll_panel(self.root, 570)
        self.left_outer.pack(side=tk.LEFT, fill=tk.Y)
        self.center = ttk.Frame(self.root, padding=8)
        self.center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.right_outer, self.right = self.scroll_panel(self.root, 560)
        self.right_outer.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(self.left, text="ROBOKINE-INDIA", font=("Arial", 22, "bold")).pack(pady=3)
        ttk.Label(self.left, text="Universal Robot Kinematics GUI | V1.1 | Revolute + Prismatic + Fixed", font=("Arial", 11, "bold")).pack()
        ttk.Label(
            self.left,
            text="Motion Axis meaning: Revolute rotates about axis | Prismatic slides along axis",
            font=("Arial", 9)
        ).pack(pady=2)
        ttk.Label(self.left, text="Developed by Dr. Priyam Parikh and Aahana Basappa", font=("Arial", 9)).pack(pady=3)

        setup = ttk.LabelFrame(self.left, text="Robot Setup", padding=8)
        setup.pack(fill=tk.X, pady=5)
        row = ttk.Frame(setup); row.pack(fill=tk.X)
        ttk.Label(row, text="Number of DOF", width=30).pack(side=tk.LEFT)
        cb = ttk.Combobox(row, textvariable=self.dof, values=[1, 2, 3, 4, 5, 6], state="readonly", width=10)
        cb.pack(side=tk.RIGHT)
        cb.bind("<<ComboboxSelected>>", lambda e: self.rebuild_joints())

        self.joint_box = ttk.LabelFrame(self.left, text="Joint Type, Motion Motion Axis, Link Length, Home, Constraints and Sliders", padding=8)
        self.joint_box.pack(fill=tk.X, pady=5)

        presets = ttk.LabelFrame(self.left, text="Robot / Joint-Type Presets", padding=8)
        presets.pack(fill=tk.X, pady=5)
        ttk.Button(presets, text="Serial Arm: R-R-R-R | Z-Y-Y-X", command=self.preset_serial_arm).pack(fill=tk.X, pady=2)
        ttk.Button(presets, text="SCARA-like: R-R-P-R | Z-Z-Z-X", command=self.preset_scara).pack(fill=tk.X, pady=2)
        ttk.Button(presets, text="Cartesian Robot: P-P-P | X-Y-Z", command=self.preset_cartesian).pack(fill=tk.X, pady=2)
        ttk.Button(presets, text="Cylindrical Robot: R-P-P | Z-Z-X", command=self.preset_cylindrical).pack(fill=tk.X, pady=2)
        ttk.Button(presets, text="All Revolute", command=self.preset_all_revolute).pack(fill=tk.X, pady=2)
        ttk.Button(presets, text="All Prismatic", command=self.preset_all_prismatic).pack(fill=tk.X, pady=2)

        home = ttk.LabelFrame(self.left, text="Home Position", padding=8)
        home.pack(fill=tk.X, pady=5)
        ttk.Button(home, text="Go to Home Position", command=self.go_home).pack(fill=tk.X, pady=2)
        ttk.Button(home, text="Set Current Sliders as Home", command=self.set_home).pack(fill=tk.X, pady=2)
        ttk.Button(home, text="Reset Home and Sliders to Zero", command=self.reset_home_zero).pack(fill=tk.X, pady=2)

        ik = ttk.LabelFrame(self.left, text="Inverse Kinematics Target", padding=8)
        ik.pack(fill=tk.X, pady=5)
        self.entry_row(ik, "Target X", self.target_x)
        self.entry_row(ik, "Target Y", self.target_y)
        self.entry_row(ik, "Target Z", self.target_z)
        self.entry_row(ik, "IK tolerance", self.ik_tol)
        self.entry_row(ik, "IK max iterations", self.ik_iter)
        self.entry_row(ik, "IK damping", self.ik_damp)
        ttk.Button(ik, text="Solve IK and Move Robot", command=self.solve_ik_button).pack(fill=tk.X, pady=2)
        ttk.Button(ik, text="Set IK Target to Current EE", command=self.set_target_current_ee).pack(fill=tk.X, pady=2)

        via = ttk.LabelFrame(self.left, text="Via Point Selection", padding=8)
        via.pack(fill=tk.X, pady=5)
        self.entry_row(via, "Via X", self.via_x)
        self.entry_row(via, "Via Y", self.via_y)
        self.entry_row(via, "Via Z", self.via_z)
        ttk.Button(via, text="Add Via Point from Fields", command=self.add_via_from_fields).pack(fill=tk.X, pady=2)
        ttk.Button(via, text="Add Current EE as Via Point", command=self.add_current_ee_as_via).pack(fill=tk.X, pady=2)
        ttk.Button(via, text="Load Selected Via to Fields", command=self.load_selected_via).pack(fill=tk.X, pady=2)
        ttk.Button(via, text="Replace Selected Via", command=self.replace_selected_via).pack(fill=tk.X, pady=2)
        ttk.Button(via, text="Move Via Up", command=self.move_via_up).pack(fill=tk.X, pady=2)
        ttk.Button(via, text="Move Via Down", command=self.move_via_down).pack(fill=tk.X, pady=2)
        ttk.Button(via, text="Delete Selected Via", command=self.delete_selected_via).pack(fill=tk.X, pady=2)
        ttk.Button(via, text="Clear All Via Points", command=self.clear_via_points).pack(fill=tk.X, pady=2)
        self.via_list = tk.Listbox(via, height=8, exportselection=False)
        self.via_list.pack(fill=tk.X, pady=3)
        self.via_list.bind("<<ListboxSelect>>", lambda e: self.on_via_select())
        ttk.Button(via, text="Save Robot Orientation Image at Selected Via", command=self.save_selected_via_image).pack(fill=tk.X, pady=2)

        ws = ttk.LabelFrame(self.left, text="Workspace Generation", padding=8)
        ws.pack(fill=tk.X, pady=5)
        self.entry_row(ws, "Workspace samples", self.ws_samples)
        ttk.Button(ws, text="Generate Workspace", command=self.generate_workspace).pack(fill=tk.X, pady=2)
        ttk.Button(ws, text="Clear Workspace", command=self.clear_workspace).pack(fill=tk.X, pady=2)
        ttk.Button(ws, text="Export Workspace CSV", command=self.export_workspace).pack(fill=tk.X, pady=2)

        sg = ttk.LabelFrame(self.left, text="Singularity Checking", padding=8)
        sg.pack(fill=tk.X, pady=5)
        self.entry_row(sg, "Rank tolerance", self.rank_tol)
        self.entry_row(sg, "Manipulability tolerance", self.manip_tol)
        self.entry_row(sg, "Condition number limit", self.cond_limit)
        ttk.Button(sg, text="Check Current Singularity", command=self.check_current_singularity).pack(fill=tk.X, pady=2)
        ttk.Button(sg, text="Scan Random Near-Singularity", command=self.scan_near_singularity).pack(fill=tk.X, pady=2)

        tr = ttk.LabelFrame(self.left, text="Trajectory Planning Through Via Points", padding=8)
        tr.pack(fill=tk.X, pady=5)
        ttk.Label(tr, text="Trajectory planning method").pack(anchor="w")
        ttk.Combobox(tr, textvariable=self.method, values=["First Order / Linear", "Second Order with Parabolic Blend", "Third Order Polynomial", "Fifth Order Polynomial"], state="readonly", width=42).pack(fill=tk.X, pady=2)
        self.entry_row(tr, "Segment time", self.segment_time)
        self.entry_row(tr, "Samples per segment", self.samples_per_segment)
        self.entry_row(tr, "Animation delay ms", self.anim_delay)
        ttk.Button(tr, text="Solve IK for Via Points", command=self.solve_all_via_ik).pack(fill=tk.X, pady=2)
        ttk.Button(tr, text="Generate Trajectory", command=self.generate_trajectory).pack(fill=tk.X, pady=2)
        ttk.Button(tr, text="Start Fast Animation", command=self.start_animation).pack(fill=tk.X, pady=2)
        ttk.Button(tr, text="Stop Animation", command=self.stop_animation).pack(fill=tk.X, pady=2)
        ttk.Button(tr, text="Resume Animation", command=self.resume_animation).pack(fill=tk.X, pady=2)
        ttk.Button(tr, text="Reset Animation", command=self.reset_animation).pack(fill=tk.X, pady=2)

        gr = ttk.LabelFrame(self.left, text="Trajectory Graph Windows", padding=8)
        gr.pack(fill=tk.X, pady=5)
        ttk.Button(gr, text="Open Joint Displacement Graph", command=self.open_displacement_graph).pack(fill=tk.X, pady=2)
        ttk.Button(gr, text="Open Joint Velocity Graph", command=self.open_velocity_graph).pack(fill=tk.X, pady=2)
        ttk.Button(gr, text="Open Joint Acceleration Graph", command=self.open_acceleration_graph).pack(fill=tk.X, pady=2)
        ttk.Button(gr, text="Open All Graphs", command=self.open_all_graphs).pack(fill=tk.X, pady=2)

        disp = ttk.LabelFrame(self.left, text="Display Options", padding=8)
        disp.pack(fill=tk.X, pady=5)
        for text, var in [("Show global XYZ axes", self.show_global_axes), ("Show joint frames", self.show_frames), ("Show rotation axes", self.show_rot_axes), ("Show dimensions", self.show_dims), ("Show workspace", self.show_workspace), ("Show via points", self.show_via), ("Show EE trajectory path", self.show_path), ("Show home pose", self.show_home)]:
            ttk.Checkbutton(disp, text=text, variable=var, command=self.update_scene).pack(anchor="w")

        ex = ttk.LabelFrame(self.left, text="Save / Export", padding=8)
        ex.pack(fill=tk.X, pady=5)
        ttk.Button(ex, text="Save Current Robot View PNG", command=self.save_current_view).pack(fill=tk.X, pady=2)
        ttk.Button(ex, text="Save Animation as GIF", command=self.save_gif).pack(fill=tk.X, pady=2)
        ttk.Button(ex, text="Save Animation as MP4", command=self.save_mp4).pack(fill=tk.X, pady=2)
        ttk.Button(ex, text="Export Numerical Output TXT", command=self.export_txt).pack(fill=tk.X, pady=2)

        self.fig_robot = plt.Figure(figsize=(8.5, 7.8), dpi=100)
        self.ax = self.fig_robot.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig_robot, master=self.center)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        ttk.Label(self.right, text="Real-Time Trajectory Planning Graphs", font=("Arial", 14, "bold")).pack(pady=4)
        self.fig_rt = plt.Figure(figsize=(5.4, 6.3), dpi=100)
        self.ax_rt_q = self.fig_rt.add_subplot(311)
        self.ax_rt_qd = self.fig_rt.add_subplot(312)
        self.ax_rt_qdd = self.fig_rt.add_subplot(313)
        self.fig_rt.tight_layout(pad=2.0)
        self.canvas_rt = FigureCanvasTkAgg(self.fig_rt, master=self.right)
        self.canvas_rt.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=4)
        ttk.Button(
            self.right,
            text="Download Real-Time Trajectory Graphs PNG",
            command=self.save_realtime_graphs
        ).pack(fill=tk.X, pady=4)
        out = ttk.LabelFrame(self.right, text="Numerical Output", padding=8)
        out.pack(fill=tk.BOTH, expand=True, pady=5)
        self.output = tk.Text(out, width=64, height=38, font=("Consolas", 9))
        self.output.pack(fill=tk.BOTH, expand=True)

    def rebuild_joints(self):
        for w in self.joint_box.winfo_children():
            w.destroy()
        self.joint_type_vars, self.slider_widgets = [], []
        self.value_label_widgets = []
        self.axis_vars, self.link_vars, self.home_vars = [], [], []
        self.min_vars, self.max_vars, self.angle_vars = [], [], []
        default_axes = ["+Z", "+Y", "+Y", "+X", "+Y", "+X"]
        h = ttk.Frame(self.joint_box); h.pack(fill=tk.X, pady=2)
        for text, width in [("Joint", 5), ("Type", 10), ("Motion Motion Axis", 12), ("Link", 8), ("Home", 8), ("Min", 8), ("Max", 8)]:
            ttk.Label(h, text=text, width=width).pack(side=tk.LEFT)
        for i in range(self.get_dof()):
            jt = tk.StringVar(value="Revolute")
            av = tk.StringVar(value=default_axes[i]); lv = tk.DoubleVar(value=100.0); hv = tk.DoubleVar(value=0.0)
            mn = tk.DoubleVar(value=-180.0); mx = tk.DoubleVar(value=180.0); ag = tk.DoubleVar(value=0.0)
            self.joint_type_vars.append(jt); self.axis_vars.append(av); self.link_vars.append(lv); self.home_vars.append(hv)
            self.min_vars.append(mn); self.max_vars.append(mx); self.angle_vars.append(ag)
            r = ttk.Frame(self.joint_box); r.pack(fill=tk.X, pady=2)
            ttk.Label(r, text=f"J{i+1}", width=5).pack(side=tk.LEFT)
            type_cb = ttk.Combobox(r, textvariable=jt, values=self.joint_type_options, state="readonly", width=9)
            type_cb.pack(side=tk.LEFT, padx=1); type_cb.bind("<<ComboboxSelected>>", lambda e, idx=i: self.on_joint_type_change(idx))
            cb = ttk.Combobox(r, textvariable=av, values=self.axis_options, state="readonly", width=7)
            cb.pack(side=tk.LEFT, padx=1); cb.bind("<<ComboboxSelected>>", lambda e: self.update_scene())
            for var in [lv, hv, mn, mx]:
                e = ttk.Entry(r, textvariable=var, width=8); e.pack(side=tk.LEFT, padx=1)
                e.bind("<Return>", lambda ev: self.update_slider_ranges_and_scene()); e.bind("<FocusOut>", lambda ev: self.update_slider_ranges_and_scene())
            value_label = ttk.Label(self.joint_box, text=f"J{i+1} current value: 0.000 deg")
            value_label.pack(anchor="w", pady=(2, 0))
            self.value_label_widgets.append(value_label)
            sl = ttk.Scale(self.joint_box, from_=-180, to=180, variable=ag, orient=tk.HORIZONTAL, command=lambda v: self.update_scene())
            sl.pack(fill=tk.X, pady=(0, 5)); self.slider_widgets.append(sl)
        self.workspace = None; self.workspace_q = None; self.workspace_err = None
        self.via_q = None; self.clear_trajectory(False); self.update_scene()

    def update_slider_ranges_and_scene(self):
        """Update slider ranges from Min/Max constraints, then refresh the robot scene."""
        self.update_slider_ranges()
        self.update_scene()

    def update_slider_ranges(self):
        """Synchronize each FK slider with its joint type and Min/Max constraints."""
        if not hasattr(self, "slider_widgets"):
            return

        for i in range(self.get_dof()):
            joint_type = self.joint_type_vars[i].get()

            if joint_type == "Fixed":
                self.slider_widgets[i].configure(from_=0, to=0)
                self.angle_vars[i].set(0.0)
                continue

            try:
                mn = float(self.min_vars[i].get())
                mx = float(self.max_vars[i].get())
            except Exception:
                continue

            if mn >= mx:
                # Keep the old range until the user corrects the constraint.
                continue

            self.slider_widgets[i].configure(from_=mn, to=mx)

            value = float(self.angle_vars[i].get())
            if value < mn:
                self.angle_vars[i].set(mn)
            elif value > mx:
                self.angle_vars[i].set(mx)

    def update_value_labels(self):
        """Show current slider value separately instead of using a confusing editable Now/MOVE column."""
        if not hasattr(self, "value_label_widgets"):
            return

        for i in range(self.get_dof()):
            unit = self.joint_unit(i)
            value = float(self.angle_vars[i].get())
            jtype = self.joint_type_vars[i].get()
            self.value_label_widgets[i].configure(
                text=f"J{i+1} current value: {value:.3f} {unit}  |  Type: {jtype}"
            )

    def on_joint_type_change(self, i):
        jt = self.joint_type_vars[i].get()
        if jt == "Revolute":
            self.min_vars[i].set(-180.0); self.max_vars[i].set(180.0); self.home_vars[i].set(0.0); self.angle_vars[i].set(0.0)
            self.slider_widgets[i].configure(from_=-180, to=180)
        elif jt == "Prismatic":
            self.min_vars[i].set(0.0); self.max_vars[i].set(150.0); self.home_vars[i].set(0.0); self.angle_vars[i].set(0.0)
            self.slider_widgets[i].configure(from_=0, to=150)
        else:
            self.min_vars[i].set(0.0); self.max_vars[i].set(0.0); self.home_vars[i].set(0.0); self.angle_vars[i].set(0.0)
            self.slider_widgets[i].configure(from_=0, to=0)
        self.workspace = None; self.via_q = None; self.clear_trajectory(False); self.update_slider_ranges(); self.update_value_labels(); self.update_scene()

    # ----------------------------- State -----------------------------
    # ----------------------------- State -----------------------------
    def get_dof(self): return int(self.dof.get())
    def get_angles(self): return np.array([v.get() for v in self.angle_vars], dtype=float)
    def set_angles(self, q):
        for i, val in enumerate(q): self.angle_vars[i].set(round(float(val), 4))
    def get_home(self): return np.array([v.get() for v in self.home_vars], dtype=float)
    def get_lower(self): return np.array([v.get() for v in self.min_vars], dtype=float)
    def get_upper(self): return np.array([v.get() for v in self.max_vars], dtype=float)

    def joint_unit(self, i):
        jt = self.joint_type_vars[i].get()
        return "deg" if jt == "Revolute" else "mm" if jt == "Prismatic" else "-"

    def joint_short(self, i):
        jt = self.joint_type_vars[i].get()
        return "R" if jt == "Revolute" else "P" if jt == "Prismatic" else "F"

    def active_indices(self):
        return [i for i in range(self.get_dof()) if self.joint_type_vars[i].get() != "Fixed"]

    def constraints_valid(self, check_current=False):
        lo, hi = self.get_lower(), self.get_upper()
        for i in range(self.get_dof()):
            if self.joint_type_vars[i].get() == "Fixed":
                continue
            if lo[i] >= hi[i]:
                messagebox.showwarning("Invalid constraints", f"J{i+1}: minimum value must be less than maximum value.")
                return False
        if check_current:
            q = self.get_angles()
            for i in range(self.get_dof()):
                if self.joint_type_vars[i].get() == "Fixed":
                    continue
                if q[i] < lo[i] or q[i] > hi[i]:
                    messagebox.showwarning("Value outside constraints", f"J{i+1} value is outside joint constraints.")
                    return False
        return True

    # ----------------------------- Kinematics -----------------------------
    # ----------------------------- Kinematics -----------------------------
    def axis_rotation(self, label, angle_deg):
        sign = -1.0 if label.startswith("-") else 1.0
        axis = label[-1]
        t = np.radians(sign * angle_deg)
        return rot_x(t) if axis == "X" else rot_y(t) if axis == "Y" else rot_z(t)

    def axis_vector_local(self, label):
        sign = -1.0 if label.startswith("-") else 1.0
        axis = label[-1]
        if axis == "X": return np.array([sign, 0, 0], dtype=float)
        if axis == "Y": return np.array([0, sign, 0], dtype=float)
        return np.array([0, 0, sign], dtype=float)

    def axis_vector_world(self, label, R):
        return R @ self.axis_vector_local(label)

    def fk(self, q=None):
        if q is None: q = self.get_angles()
        T = np.eye(4)
        frames = [("F0", T.copy())]
        axes = []
        for i in range(self.get_dof()):
            label = self.axis_vars[i].get(); L = self.link_vars[i].get(); jt = self.joint_type_vars[i].get(); val = q[i]
            origin = T[:3, 3].copy(); axis_w = self.axis_vector_world(label, T[:3, :3])
            axes.append((origin, axis_w, label, jt, f"J{i+1}"))
            if jt == "Revolute":
                T = T @ tf(self.axis_rotation(label, val), [0, 0, 0])
                frames.append((f"After R J{i+1}", T.copy()))
            elif jt == "Prismatic":
                T = T @ tf(np.eye(3), self.axis_vector_local(label) * val)
                frames.append((f"After P J{i+1}", T.copy()))
            else:
                frames.append((f"Fixed J{i+1}", T.copy()))
            T = T @ tf(np.eye(3), [L, 0, 0])
            frames.append((f"F{i+1}", T.copy()))
        return frames, axes, T

    def points_from_frames(self, frames):
        return np.array([frames[0][1][:3, 3]] + [frames[2*i+2][1][:3, 3] for i in range(self.get_dof())])

    def ee_position(self, q=None):
        return self.fk(q)[2][:3, 3]

    def jacobian(self, q):
        q = np.array(q, dtype=float)
        active = self.active_indices()
        J = np.zeros((3, len(active)))
        p0 = self.ee_position(q)
        for col, i in enumerate(active):
            qp = q.copy()
            if self.joint_type_vars[i].get() == "Revolute":
                delta = 0.05
                qp[i] += delta
                J[:, col] = (self.ee_position(qp) - p0) / np.radians(delta)
            else:
                delta = 0.05
                qp[i] += delta
                J[:, col] = (self.ee_position(qp) - p0) / delta
        return J

    def numerical_ik(self, target, initial=None):
        if not self.constraints_valid(False): return None, None
        active = self.active_indices()
        if not active: return None, None
        target = np.array(target, dtype=float)
        q = self.get_home().copy() if initial is None else np.array(initial, dtype=float)
        lo, hi = self.get_lower(), self.get_upper(); q = np.clip(q, lo, hi)
        tol = max(float(self.ik_tol.get()), 1e-6); max_iter = max(int(self.ik_iter.get()), 10); damping = max(float(self.ik_damp.get()), 1e-6)
        for _ in range(max_iter):
            err = target - self.ee_position(q); norm = float(np.linalg.norm(err))
            if norm <= tol: return q, norm
            J = self.jacobian(q); JJt = J @ J.T
            try:
                dq = J.T @ np.linalg.solve(JJt + damping*damping*np.eye(3), err)
            except np.linalg.LinAlgError:
                return None, None
            for k, i in enumerate(active):
                if self.joint_type_vars[i].get() == "Revolute":
                    step = np.degrees(dq[k]); step = np.clip(step, -5, 5)
                else:
                    step = dq[k]; step = np.clip(step, -5, 5)
                q[i] = np.clip(q[i] + 0.85*step, lo[i], hi[i])
        final_err = float(np.linalg.norm(target - self.ee_position(q)))
        return (q, final_err) if final_err <= max(3*tol, tol+0.5) else (None, final_err)

    def solve_ik_button(self):
        target = np.array([self.target_x.get(), self.target_y.get(), self.target_z.get()])
        q, err = self.numerical_ik(target, self.get_angles())
        if q is None:
            messagebox.showwarning("IK failed", f"Target could not be solved. Final error: {err}")
            return
        self.set_angles(q); self.clear_trajectory(False); self.update_scene()
        self.append_output(f"IK solved for target {target}. Error={err:.6f}\n" + ", ".join([f"J{i+1}={q[i]:.3f}°" for i in range(len(q))]) + "\n\n")

    def set_target_current_ee(self):
        p = self.ee_position(self.get_angles())
        self.target_x.set(round(float(p[0]), 4)); self.target_y.set(round(float(p[1]), 4)); self.target_z.set(round(float(p[2]), 4))
        self.update_scene()

    # ----------------------------- Via points -----------------------------
    def refresh_via_list(self):
        self.via_list.delete(0, tk.END)
        for i, p in enumerate(self.via_points):
            self.via_list.insert(tk.END, f"P{i+1}: X={p[0]:.2f}, Y={p[1]:.2f}, Z={p[2]:.2f}")

    def on_via_select(self):
        sel = self.via_list.curselection()
        self.selected_via = int(sel[0]) if sel else None

    def add_via_from_fields(self):
        self.via_points.append(np.array([self.via_x.get(), self.via_y.get(), self.via_z.get()], dtype=float))
        self.via_q = None; self.refresh_via_list(); self.update_scene()

    def add_current_ee_as_via(self):
        self.via_points.append(self.ee_position(self.get_angles()))
        self.via_q = None; self.refresh_via_list(); self.update_scene()

    def load_selected_via(self):
        sel = self.via_list.curselection()
        if not sel: messagebox.showwarning("No via selected", "Select a via point first."); return
        p = self.via_points[int(sel[0])]
        self.via_x.set(round(float(p[0]), 4)); self.via_y.set(round(float(p[1]), 4)); self.via_z.set(round(float(p[2]), 4))

    def replace_selected_via(self):
        sel = self.via_list.curselection()
        if not sel: messagebox.showwarning("No via selected", "Select a via point first."); return
        i = int(sel[0]); self.via_points[i] = np.array([self.via_x.get(), self.via_y.get(), self.via_z.get()], dtype=float)
        self.via_q = None; self.refresh_via_list(); self.via_list.selection_set(i); self.update_scene()

    def move_via_up(self):
        sel = self.via_list.curselection()
        if not sel: return
        i = int(sel[0])
        if i <= 0: return
        self.via_points[i-1], self.via_points[i] = self.via_points[i], self.via_points[i-1]
        self.via_q = None; self.refresh_via_list(); self.via_list.selection_set(i-1); self.update_scene()

    def move_via_down(self):
        sel = self.via_list.curselection()
        if not sel: return
        i = int(sel[0])
        if i >= len(self.via_points)-1: return
        self.via_points[i+1], self.via_points[i] = self.via_points[i], self.via_points[i+1]
        self.via_q = None; self.refresh_via_list(); self.via_list.selection_set(i+1); self.update_scene()

    def delete_selected_via(self):
        sel = self.via_list.curselection()
        if not sel: messagebox.showwarning("No via selected", "Select a via point first."); return
        del self.via_points[int(sel[0])]
        self.via_q = None; self.refresh_via_list(); self.update_scene()

    def clear_via_points(self):
        self.via_points = []; self.via_q = None; self.clear_trajectory(False); self.refresh_via_list(); self.update_scene()

    # ----------------------------- Workspace -----------------------------
    def generate_workspace(self):
        if not self.constraints_valid(False): return
        n = min(max(int(self.ws_samples.get()), 100), 50000)
        lo, hi = self.get_lower(), self.get_upper()
        rng = np.random.default_rng(42)
        Q = np.zeros((n, self.get_dof()))
        for i in range(self.get_dof()):
            if self.joint_type_vars[i].get() == "Fixed":
                Q[:, i] = 0.0
            else:
                Q[:, i] = rng.uniform(lo[i], hi[i], size=n)
        P = np.array([self.ee_position(q) for q in Q])
        target = np.array([self.target_x.get(), self.target_y.get(), self.target_z.get()])
        E = np.linalg.norm(P - target[None, :], axis=1)
        self.workspace, self.workspace_q, self.workspace_err = P, Q, E
        self.show_workspace.set(True); self.update_scene()
        idx = int(np.argmin(E))
        self.append_output(f"Workspace generated: {n} samples\nX range {P[:,0].min():.2f} to {P[:,0].max():.2f}\nY range {P[:,1].min():.2f} to {P[:,1].max():.2f}\nZ range {P[:,2].min():.2f} to {P[:,2].max():.2f}\nNearest error to IK target = {E[idx]:.4f}\n\n")

    def clear_workspace(self):
        self.workspace = self.workspace_q = self.workspace_err = None; self.show_workspace.set(False); self.update_scene()

    def export_workspace(self):
        if self.workspace is None:
            messagebox.showwarning("No workspace", "Generate workspace first."); return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV file", "*.csv")])
        if path:
            np.savetxt(path, np.column_stack([self.workspace, self.workspace_err]), delimiter=",", header="X,Y,Z,Distance_To_Target", comments="")
            messagebox.showinfo("Saved", f"Workspace CSV saved:\n{path}")

    # ----------------------------- Singularity -----------------------------
    def singularity_metrics(self, q=None):
        q = self.get_angles() if q is None else q
        J = self.jacobian(q)
        rank = int(np.linalg.matrix_rank(J, tol=max(float(self.rank_tol.get()), 1e-12))) if J.size else 0
        s = np.linalg.svd(J, compute_uv=False) if J.size else np.array([])
        min_sv, max_sv = (float(np.min(s)), float(np.max(s))) if len(s) else (0.0, 0.0)
        cond = np.inf if min_sv < 1e-12 else max_sv / min_sv
        detv = float(np.linalg.det(J @ J.T)) if J.size else 0.0
        if detv < 0 and abs(detv) < 1e-8: detv = 0.0
        manip = float(np.sqrt(max(detv, 0.0)))
        return {"J": J, "rank": rank, "singular_values": s, "condition_number": cond, "manipulability": manip, "active_count": len(self.active_indices())}

    def singularity_status(self, m):
        req = min(3, m.get("active_count", self.get_dof()))
        if req == 0: return "NO ACTIVE JOINTS: Motion is not possible."
        if m["rank"] < req: return "SINGULAR: Jacobian rank is deficient."
        if m["manipulability"] < max(float(self.manip_tol.get()), 1e-12): return "NEAR SINGULAR: Manipulability is very low."
        if not np.isfinite(m["condition_number"]): return "NEAR SINGULAR: Condition number is infinite."
        if m["condition_number"] > max(float(self.cond_limit.get()), 1e-12): return "NEAR SINGULAR: Condition number is high."
        return "REGULAR: Not singular for translational position task."

    def check_current_singularity(self):
        m = self.singularity_metrics(); status = self.singularity_status(m); self.draw_singular_values(m)
        self.append_output(f"Singularity check\nRequired rank={min(3,self.get_dof())}\nRank={m['rank']}\nManipulability={m['manipulability']:.8f}\nCondition={m['condition_number']:.6f}\nSingular values={np.array2string(m['singular_values'], precision=5)}\nStatus={status}\n\n")
        (messagebox.showwarning if status.startswith(("SINGULAR", "NEAR")) else messagebox.showinfo)("Singularity Status", status)

    def scan_near_singularity(self):
        if not self.constraints_valid(False): return
        n = min(max(int(self.ws_samples.get()), 100), 10000)
        lo, hi = self.get_lower(), self.get_upper(); rng = np.random.default_rng(123)
        best_q, best_m, best_manip = None, None, np.inf
        for _ in range(n):
            q = self.get_home().copy()
            for ii in range(self.get_dof()):
                if self.joint_type_vars[ii].get() != "Fixed":
                    q[ii] = rng.uniform(lo[ii], hi[ii])
            m = self.singularity_metrics(q)
            if m["manipulability"] < best_manip: best_q, best_m, best_manip = q, m, m["manipulability"]
        if best_q is not None:
            self.set_angles(best_q); self.update_scene(); self.draw_singular_values(best_m)
            self.append_output(f"Near-singularity scan: lowest manipulability={best_manip:.8f}\nAngles=" + ", ".join([f"J{i+1}={best_q[i]:.3f}°" for i in range(len(best_q))]) + f"\nStatus={self.singularity_status(best_m)}\n\n")

    def draw_singular_values(self, m):
        # Singularity graph removed. Singularity values remain in numerical output.
        self.append_output(
            "Singularity graph removed. Rank, manipulability, condition number, "
            "and singular values are available in the numerical output.\n\n"
        )


    # ----------------------------- Trajectory -----------------------------
    def solve_all_via_ik(self):
        if not self.via_points:
            messagebox.showwarning("No via points", "Add at least one via point."); return None
        qs = []; q_guess = self.get_home().copy(); self.append_output("Solving IK for via points...\n")
        for i, p in enumerate(self.via_points):
            q, err = self.numerical_ik(p, q_guess)
            if q is None:
                messagebox.showwarning("IK failed", f"IK failed for P{i+1}. Error: {err}"); return None
            qs.append(q); q_guess = q.copy(); self.append_output(f"P{i+1}: error={err:.6f}; " + ", ".join([f"J{j+1}={q[j]:.3f}°" for j in range(len(q))]) + "\n")
        self.via_q = np.array(qs); self.append_output("All via IK solved.\n\n"); return self.via_q

    def profile(self, q0, q1, T, N):
        t = np.linspace(0, T, N); tau = t / T; dq = q1 - q0; method = self.method.get()
        if method == "First Order / Linear":
            s = tau; sd = np.ones_like(tau) / T; sdd = np.zeros_like(tau)
        elif method == "Second Order with Parabolic Blend":
            tb = 0.25*T; v = 1.0/(T-tb); a = v/tb; s=np.zeros(N); sd=np.zeros(N); sdd=np.zeros(N)
            for i, ti in enumerate(t):
                if ti < tb: s[i]=0.5*a*ti**2; sd[i]=a*ti; sdd[i]=a
                elif ti <= T-tb: s[i]=v*(ti-tb/2); sd[i]=v; sdd[i]=0
                else: dt=T-ti; s[i]=1-0.5*a*dt**2; sd[i]=a*dt; sdd[i]=-a
        elif method == "Third Order Polynomial":
            s = 3*tau**2 - 2*tau**3; sd = (6*tau - 6*tau**2)/T; sdd = (6 - 12*tau)/(T*T)
        else:
            s = 10*tau**3 - 15*tau**4 + 6*tau**5; sd = (30*tau**2 - 60*tau**3 + 30*tau**4)/T; sdd = (60*tau - 180*tau**2 + 120*tau**3)/(T*T)
        return t, q0 + s[:,None]*dq, sd[:,None]*dq, sdd[:,None]*dq

    def generate_trajectory(self):
        if not self.via_points:
            messagebox.showwarning("No via points", "Add at least one via point."); return
        if self.via_q is None and self.solve_all_via_ik() is None: return
        way = [self.get_home()] + list(self.via_q)
        Tseg = float(self.segment_time.get()); Nseg = int(self.samples_per_segment.get())
        if Tseg <= 0 or Nseg < 10:
            messagebox.showwarning("Invalid trajectory settings", "Segment time must be >0 and samples per segment must be >=10."); return
        all_t=[]; all_q=[]; all_qd=[]; all_qdd=[]; seg=[]; offset=0.0
        for i in range(len(way)-1):
            t,q,qd,qdd = self.profile(np.array(way[i]), np.array(way[i+1]), Tseg, Nseg); t += offset
            if i > 0: t,q,qd,qdd = t[1:], q[1:], qd[1:], qdd[1:]
            all_t.append(t); all_q.append(q); all_qd.append(qd); all_qdd.append(qdd); seg.extend([i]*len(t)); offset += Tseg
        q_total = np.vstack(all_q); lo,hi = self.get_lower(), self.get_upper()
        for ii in range(self.get_dof()):
            if self.joint_type_vars[ii].get() != "Fixed" and (np.any(q_total[:, ii] < lo[ii]) or np.any(q_total[:, ii] > hi[ii])):
                messagebox.showwarning("Constraint violation", f"Trajectory violates J{ii+1} constraints."); return
        self.traj_t = np.concatenate(all_t); self.traj_q=q_total; self.traj_qd=np.vstack(all_qd); self.traj_qdd=np.vstack(all_qdd); self.traj_seg=np.array(seg)
        self.ee_path = np.array([self.ee_position(q) for q in self.traj_q]); self.frame=0; self.set_angles(self.traj_q[0]); self.update_scene()
        self.append_output(f"Trajectory generated\nMethod={self.method.get()}\nSegments={len(way)-1}\nTotal time={self.traj_t[-1]:.3f} s\nSamples={len(self.traj_t)}\n\n")

    def clear_trajectory(self, clear_output=False):
        self.traj_t = self.traj_q = self.traj_qd = self.traj_qdd = self.traj_seg = self.ee_path = None; self.frame = 0
        if hasattr(self, "ax_rt_q"): self.update_realtime_graphs(None)
        if clear_output and hasattr(self, "output"): self.output.delete("1.0", tk.END)

    # ----------------------------- Animation -----------------------------
    def animation_step(self):
        if not self.running or self.traj_q is None: return
        if self.frame >= len(self.traj_q): self.running = False; return
        q = self.traj_q[self.frame]; self.set_angles(q); self.update_scene(q); self.update_realtime_graphs(self.frame); self.frame += 1
        self.after_id = self.root.after(max(1, int(self.anim_delay.get())), self.animation_step)

    def start_animation(self):
        if self.traj_q is None: self.generate_trajectory()
        if self.traj_q is None: return
        self.stop_animation(); self.frame=0; self.running=True; self.animation_step()

    def stop_animation(self):
        self.running = False
        if self.after_id is not None:
            try: self.root.after_cancel(self.after_id)
            except Exception: pass
            self.after_id = None

    def resume_animation(self):
        if self.traj_q is None: return
        if self.frame >= len(self.traj_q): self.frame=0
        self.running=True; self.animation_step()

    def reset_animation(self):
        self.stop_animation(); self.frame=0
        if self.traj_q is not None: self.set_angles(self.traj_q[0]); self.update_scene(self.traj_q[0]); self.update_realtime_graphs(0)
        else: self.go_home()

    # ----------------------------- Real-time embedded trajectory graphs -----------------------------
    def update_realtime_graphs(self, current_index=None):
        if not hasattr(self, "ax_rt_q"):
            return

        axes = [self.ax_rt_q, self.ax_rt_qd, self.ax_rt_qdd]

        if self.traj_t is None or self.traj_q is None:
            for ax in axes:
                ax.clear()
                ax.grid(True)
            self.ax_rt_q.set_title("Joint Displacement")
            self.ax_rt_qd.set_title("Joint Velocity")
            self.ax_rt_qdd.set_title("Joint Acceleration")
            self.canvas_rt.draw_idle()
            return

        graphs = [
            (self.ax_rt_q, self.traj_q, "Joint Displacement", "deg / mm"),
            (self.ax_rt_qd, self.traj_qd, "Joint Velocity", "deg/s or mm/s"),
            (self.ax_rt_qdd, self.traj_qdd, "Joint Acceleration", "deg/s² or mm/s²")
        ]

        for ax, data, title, ylabel in graphs:
            ax.clear()
            for j in range(data.shape[1]):
                ax.plot(self.traj_t, data[:, j], linewidth=1.4, label=f"J{j+1} ({self.joint_short(j)})")

            if current_index is not None and 0 <= current_index < len(self.traj_t):
                ax.axvline(
                    self.traj_t[current_index],
                    color="black",
                    linestyle="--",
                    linewidth=1.5
                )

            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(ylabel)
            ax.grid(True)
            ax.legend(fontsize=7, loc="best")

        self.fig_rt.tight_layout(pad=2.0)
        self.canvas_rt.draw_idle()

    def save_realtime_graphs(self):
        if not hasattr(self, "fig_rt"):
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")]
        )
        if not path:
            return

        self.fig_rt.savefig(path, dpi=220, bbox_inches="tight")
        messagebox.showinfo("Saved", f"Real-time trajectory graphs saved:\n{path}")

    # ----------------------------- Graphs -----------------------------
    def graph_window(self, data, title, ylabel):
        if self.traj_t is None or data is None:
            messagebox.showwarning("No trajectory", "Generate trajectory first."); return
        win = tk.Toplevel(self.root); win.title(title); win.geometry("900x650")
        fig = plt.Figure(figsize=(8.5, 5.8), dpi=100); ax = fig.add_subplot(111)
        for j in range(data.shape[1]): ax.plot(self.traj_t, data[:,j], linewidth=1.7, label=f"J{j+1} ({self.joint_short(j)})")
        ax.set_title(title); ax.set_xlabel("Time (s)"); ax.set_ylabel(ylabel); ax.grid(True); ax.legend()
        canvas = FigureCanvasTkAgg(fig, master=win); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        def save():
            path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG image", "*.png")])
            if path: fig.savefig(path, dpi=220, bbox_inches="tight"); messagebox.showinfo("Saved", f"Graph saved:\n{path}")
        ttk.Button(win, text="Download / Save Graph PNG", command=save).pack(fill=tk.X, padx=8, pady=6); canvas.draw_idle()

    def open_displacement_graph(self): self.graph_window(self.traj_q, "Joint Displacement Graph", "deg / mm")
    def open_velocity_graph(self): self.graph_window(self.traj_qd, "Joint Velocity Graph", "deg/s or mm/s")
    def open_acceleration_graph(self): self.graph_window(self.traj_qdd, "Joint Acceleration Graph", "deg/s² or mm/s²")
    def open_all_graphs(self): self.open_displacement_graph(); self.open_velocity_graph(); self.open_acceleration_graph()

    # ----------------------------- Drawing -----------------------------
    def draw_axes_at(self, T, label, L):
        if not self.show_frames.get(): return
        p=T[:3,3]; R=T[:3,:3]
        for vec,col in [(R@np.array([L,0,0]),self.colors['x']), (R@np.array([0,L,0]),self.colors['y']), (R@np.array([0,0,L]),self.colors['z'])]:
            self.ax.quiver(p[0],p[1],p[2],vec[0],vec[1],vec[2],color=col,linewidth=1.4,arrow_length_ratio=0.18)
        self.ax.text(p[0],p[1],p[2],label,fontsize=8,fontweight='bold')

    def draw_motion_axis(self, origin, axis, label, L, joint_type="Revolute"):
        if not self.show_rot_axes.get() or joint_type == "Fixed": return
        axis=np.array(axis,dtype=float); n=np.linalg.norm(axis)
        if n<1e-9: return
        axis=axis/n; v=axis*L
        if joint_type == "Prismatic":
            col = "#008B8B"; axis_name = f"{label} P-axis"
        else:
            col = self.colors['x'] if abs(axis[0])>=abs(axis[1]) and abs(axis[0])>=abs(axis[2]) else self.colors['y'] if abs(axis[1])>=abs(axis[2]) else self.colors['z']
            axis_name = f"{label} R-axis"
        p=origin
        self.ax.quiver(p[0],p[1],p[2],v[0],v[1],v[2],color=col,linewidth=4,arrow_length_ratio=0.16)
        if joint_type == "Revolute":
            self.ax.quiver(p[0],p[1],p[2],-v[0],-v[1],-v[2],color=col,linewidth=4,arrow_length_ratio=0.16)
        self.ax.text(p[0]+1.08*v[0],p[1]+1.08*v[1],p[2]+1.08*v[2],axis_name,color=col,fontsize=8,fontweight='bold')

    def update_scene(self, q=None):
        if not hasattr(self, "ax"): return
        self.update_slider_ranges()
        self.update_value_labels()
        self.ax.clear(); q = self.get_angles() if q is None else q
        frames, axes, T = self.fk(q); pts = self.points_from_frames(frames); ee=T[:3,3]
        total=sum(abs(v.get()) for v in self.link_vars)
        pris=sum(max(abs(self.min_vars[i].get()), abs(self.max_vars[i].get())) for i in range(self.get_dof()) if self.joint_type_vars[i].get()=="Prismatic")
        lim=max(100.0,(total+pris)*0.75); zlim=max(100.0,(total+pris)*0.75)
        if self.workspace is not None and self.show_workspace.get():
            wp=self.workspace; idx=np.linspace(0,len(wp)-1,min(len(wp),8000)).astype(int)
            self.ax.scatter(wp[idx,0],wp[idx,1],wp[idx,2],s=4,alpha=0.25,color=self.colors['workspace'],label='Workspace')
        if self.show_home.get():
            hp=self.points_from_frames(self.fk(self.get_home())[0])
            self.ax.plot(hp[:,0],hp[:,1],hp[:,2],color=self.colors['home'],linestyle=':',marker='o',linewidth=2.5,alpha=0.7,label='Home pose')
        self.ax.plot(pts[:,0],pts[:,1],pts[:,2],color=self.colors['robot'],marker='o',linewidth=5,markersize=8,label='Robot pose')
        for i,p in enumerate(pts[:-1]):
            color = "#008B8B" if self.joint_type_vars[i].get()=="Prismatic" else "#555555" if self.joint_type_vars[i].get()=="Fixed" else self.colors['joint']
            self.ax.scatter(p[0],p[1],p[2],color=color,s=90); self.ax.text(p[0],p[1],p[2],f"J{i+1}-{self.joint_short(i)}",fontsize=8,fontweight='bold')
        self.ax.scatter(ee[0],ee[1],ee[2],color=self.colors['ee'],s=130,label='End Effector'); self.ax.text(ee[0],ee[1],ee[2],'EE',color=self.colors['ee'],fontsize=10,fontweight='bold')
        if self.via_points and self.show_via.get():
            via=np.array(self.via_points); self.ax.scatter(via[:,0],via[:,1],via[:,2],marker='*',color=self.colors['via'],s=160,label='Via points')
            for i,p in enumerate(via): self.ax.text(p[0],p[1],p[2],f"P{i+1}",color=self.colors['via'],fontsize=9,fontweight='bold')
        if self.ee_path is not None and self.show_path.get():
            upto = self.frame if self.frame>1 else len(self.ee_path)
            self.ax.plot(self.ee_path[:upto,0],self.ee_path[:upto,1],self.ee_path[:upto,2],color=self.colors['path'],linewidth=2.3,label='EE trajectory')
        if self.show_global_axes.get():
            a=lim*0.25
            self.ax.quiver(0,0,0,a,0,0,color=self.colors['x'],linewidth=3,arrow_length_ratio=0.12); self.ax.text(a,0,0,'X0',color=self.colors['x'],fontweight='bold')
            self.ax.quiver(0,0,0,0,a,0,color=self.colors['y'],linewidth=3,arrow_length_ratio=0.12); self.ax.text(0,a,0,'Y0',color=self.colors['y'],fontweight='bold')
            self.ax.quiver(0,0,0,0,0,a,color=self.colors['z'],linewidth=3,arrow_length_ratio=0.12); self.ax.text(0,0,a,'Z0',color=self.colors['z'],fontweight='bold')
        self.draw_axes_at(frames[0][1], 'F0', lim*0.08)
        for i in range(self.get_dof()): self.draw_axes_at(frames[2*i+2][1], f"F{i+1}", lim*0.08)
        for origin, axis, axis_label, joint_type, label in axes: self.draw_motion_axis(origin, axis, label, lim*0.11, joint_type)
        if self.show_dims.get():
            for i in range(len(pts)-1):
                p1,p2=pts[i],pts[i+1]; self.ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],'--',color='#555555',linewidth=1.2)
                mid=(p1+p2)/2; self.ax.text(mid[0],mid[1],mid[2],f"L{i+1}={self.link_vars[i].get():.1f}",fontsize=8,color='#333333')
        title="ROBOKINE-INDIA V1.1 | Mixed Revolute-Prismatic-Fixed Joint GUI"
        if self.traj_t is not None and 0<=self.frame<len(self.traj_t): title += f"\nTrajectory time = {self.traj_t[self.frame]:.3f} s"
        self.ax.set_xlim(-lim,lim); self.ax.set_ylim(-lim,lim); self.ax.set_zlim(-zlim*0.4,zlim)
        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z'); self.ax.set_title(title)
        self.ax.text2D(0.02,0.02,"V1.1 | Developed by Dr. Priyam Parikh and Aahanan Basappa",transform=self.ax.transAxes,fontsize=8)
        self.ax.grid(True); self.ax.legend(fontsize=7,loc='upper right'); self.ax.view_init(elev=25,azim=45); self.canvas.draw_idle()
        self.print_live_output(q,T)

    def print_live_output(self,q,T):
        if not hasattr(self,'output'): return
        ee=T[:3,3]; m=self.singularity_metrics(q); status=self.singularity_status(m)
        self.output.delete('1.0',tk.END)
        self.output.insert(tk.END,"ROBOKINE-INDIA V1.1\nDeveloped by Dr. Priyam Parikh and Aahanan Basappa\n"+'='*70+'\n\n')
        self.output.insert(tk.END,f"Selected DOF = {self.get_dof()}\n\nCurrent joint configuration:\n")
        for i in range(self.get_dof()):
            self.output.insert(tk.END,f"J{i+1}: type={self.joint_type_vars[i].get():10s}, axis={self.axis_vars[i].get():>2s}, value={q[i]:9.3f} {self.joint_unit(i):3s}, L={self.link_vars[i].get():8.3f}, limit=[{self.min_vars[i].get():.1f}, {self.max_vars[i].get():.1f}]\n")
        self.output.insert(tk.END,f"\nEnd-effector position:\nX={ee[0]:10.4f}\nY={ee[1]:10.4f}\nZ={ee[2]:10.4f}\n\nT0_EE:\n")
        for row in T: self.output.insert(tk.END,"[ "+"  ".join([f"{v: .5f}" for v in row])+" ]\n")
        self.output.insert(tk.END,f"\nSingularity:\nRank={m['rank']} / {min(3,m.get('active_count', self.get_dof()))}\nManipulability={m['manipulability']:.8f}\nCondition number={m['condition_number']:.6f}\nStatus={status}\n")
        if self.traj_t is not None:
            self.output.insert(tk.END,f"\nTrajectory:\nMethod={self.method.get()}\nTotal time={self.traj_t[-1]:.4f} s\nSamples={len(self.traj_t)}\n")

    # ----------------------------- Home and export -----------------------------
    def apply_preset(self, types, axes, links=None):
        for i in range(self.get_dof()):
            self.joint_type_vars[i].set(types[i % len(types)])
            self.axis_vars[i].set(axes[i % len(axes)])
            if links is not None:
                self.link_vars[i].set(links[i % len(links)])
            self.on_joint_type_change(i)
        self.update_scene()

    def preset_serial_arm(self):
        self.apply_preset(["Revolute"], ["+Z", "+Y", "+Y", "+X", "+Y", "+X"])

    def preset_scara(self):
        self.apply_preset(["Revolute", "Revolute", "Prismatic", "Revolute", "Fixed", "Fixed"], ["+Z", "+Z", "+Z", "+X", "+X", "+X"])

    def preset_cartesian(self):
        self.apply_preset(["Prismatic", "Prismatic", "Prismatic", "Fixed", "Fixed", "Fixed"], ["+X", "+Y", "+Z", "+X", "+X", "+X"], [0, 0, 0, 50, 50, 50])

    def preset_cylindrical(self):
        self.apply_preset(["Revolute", "Prismatic", "Prismatic", "Fixed", "Fixed", "Fixed"], ["+Z", "+Z", "+X", "+X", "+X", "+X"], [50, 0, 0, 50, 50, 50])

    def preset_all_revolute(self):
        self.apply_preset(["Revolute"], ["+Z"])

    def preset_all_prismatic(self):
        self.apply_preset(["Prismatic"], ["+X", "+Y", "+Z"])

    def go_home(self):
        self.stop_animation(); q=self.get_home(); lo,hi=self.get_lower(),self.get_upper()
        for i in range(self.get_dof()):
            if self.joint_type_vars[i].get() != "Fixed" and (q[i] < lo[i] or q[i] > hi[i]):
                messagebox.showwarning("Home outside limits", f"J{i+1} home value is outside joint constraints."); return
        self.clear_trajectory(False); self.set_angles(q); self.update_scene(q)
    def set_home(self):
        for i,val in enumerate(self.get_angles()): self.home_vars[i].set(round(float(val),4))
        self.append_output("Current pose saved as Home Position.\n\n")
    def reset_home_zero(self):
        for i in range(self.get_dof()): self.home_vars[i].set(0.0); self.angle_vars[i].set(0.0)
        self.clear_trajectory(False); self.update_scene()

    def append_output(self, text):
        if hasattr(self,'output'):
            self.output.insert(tk.END,text); self.output.see(tk.END)
    def save_current_view(self):
        path=filedialog.asksaveasfilename(defaultextension='.png',filetypes=[('PNG image','*.png')])
        if path: self.fig_robot.savefig(path,dpi=220,bbox_inches='tight'); messagebox.showinfo('Saved',f'Robot view saved:\n{path}')
    def save_selected_via_image(self):
        sel=self.via_list.curselection()
        if not sel: messagebox.showwarning('No via selected','Select a via point first.'); return
        i=int(sel[0])
        if self.via_q is None and self.solve_all_via_ik() is None: return
        if i>=len(self.via_q): messagebox.showwarning('Invalid via','No IK solution for selected via point.'); return
        backup=self.get_angles().copy(); self.set_angles(self.via_q[i]); self.update_scene(self.via_q[i])
        path=filedialog.asksaveasfilename(defaultextension='.png',initialfile=f'ROBOKINE_INDIA_Via_{i+1}_Orientation.png',filetypes=[('PNG image','*.png')])
        if path: self.fig_robot.savefig(path,dpi=220,bbox_inches='tight'); messagebox.showinfo('Saved',f'Via orientation image saved:\n{path}')
        self.set_angles(backup); self.update_scene(backup)
    def export_txt(self):
        path=filedialog.asksaveasfilename(defaultextension='.txt',filetypes=[('Text file','*.txt')])
        if path:
            with open(path,'w',encoding='utf-8') as f: f.write(self.output.get('1.0',tk.END))
            messagebox.showinfo('Saved',f'Output saved:\n{path}')

    def save_anim_common(self,path,writer):
        if self.traj_q is None: self.generate_trajectory()
        if self.traj_q is None: return
        fig=plt.Figure(figsize=(8,7),dpi=120); ax=fig.add_subplot(111,projection='3d')
        total=sum(abs(v.get()) for v in self.link_vars); pris=sum(max(abs(self.min_vars[i].get()), abs(self.max_vars[i].get())) for i in range(self.get_dof()) if self.joint_type_vars[i].get()=="Prismatic"); lim=max(100.0,(total+pris)*0.75); zlim=max(100.0,(total+pris)*0.75); via=np.array(self.via_points) if self.via_points else None
        def upd(k):
            ax.clear(); frames,_,_=self.fk(self.traj_q[k]); pts=self.points_from_frames(frames)
            ax.plot(pts[:,0],pts[:,1],pts[:,2],color=self.colors['robot'],marker='o',linewidth=5,markersize=8); ee=pts[-1]; ax.scatter(ee[0],ee[1],ee[2],color=self.colors['ee'],s=120)
            if via is not None:
                ax.scatter(via[:,0],via[:,1],via[:,2],marker='*',color=self.colors['via'],s=150)
                for i,p in enumerate(via): ax.text(p[0],p[1],p[2],f"P{i+1}",color=self.colors['via'],fontsize=8,fontweight='bold')
            if self.ee_path is not None: ax.plot(self.ee_path[:k+1,0],self.ee_path[:k+1,1],self.ee_path[:k+1,2],color=self.colors['path'],linewidth=2.2)
            a=lim*0.25; ax.quiver(0,0,0,a,0,0,color=self.colors['x'],linewidth=3,arrow_length_ratio=0.12); ax.quiver(0,0,0,0,a,0,color=self.colors['y'],linewidth=3,arrow_length_ratio=0.12); ax.quiver(0,0,0,0,0,a,color=self.colors['z'],linewidth=3,arrow_length_ratio=0.12)
            ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-zlim*0.4,zlim); ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.set_title(f"ROBOKINE-INDIA V1.1 | t={self.traj_t[k]:.2f} s"); ax.text2D(0.02,0.02,"V1.1 | Developed by Dr. Priyam Parikh and Aahanan Basappa",transform=ax.transAxes,fontsize=8); ax.grid(True); ax.view_init(elev=25,azim=45+k*0.08)
        step=max(1,len(self.traj_q)//180); anim=FuncAnimation(fig,upd,frames=list(range(0,len(self.traj_q),step)),interval=35,repeat=False); anim.save(path,writer=writer)
    def save_gif(self):
        path=filedialog.asksaveasfilename(defaultextension='.gif',filetypes=[('GIF','*.gif')])
        if path: self.save_anim_common(path,PillowWriter(fps=20)); messagebox.showinfo('Saved',f'GIF saved:\n{path}')
    def save_mp4(self):
        path=filedialog.asksaveasfilename(defaultextension='.mp4',filetypes=[('MP4','*.mp4')])
        if path:
            try: self.save_anim_common(path,FFMpegWriter(fps=25)); messagebox.showinfo('Saved',f'MP4 saved:\n{path}')
            except Exception as e: messagebox.showerror('MP4 export error','MP4 export requires FFmpeg installed and added to PATH.\n\n'+str(e))


if __name__ == '__main__':
    root = tk.Tk()
    app = RoboKineIndia(root)
    root.mainloop()
