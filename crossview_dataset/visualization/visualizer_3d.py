from vispy import scene
from vispy.scene import cameras, visuals
from vispy.visuals.line import LineVisual

from threading import Thread
from multiprocessing import Process, Queue
from queue import Full
import time
import numpy as np
import copy


class XYZAxisVisual(LineVisual):
    """
    Simple 3D axis for indicating coordinate system orientation. Axes are
    x=red, y=green, z=blue.
    """

    def __init__(self, world_ltrb, max_z, **kwargs):
        x1, y1, x2, y2 = world_ltrb

        verts = np.array(
            [
                [x1, y1, 0],
                [x2 + 10, y1, 0],
                [x1, y1, 0],
                [x1, y2 + 10, 0],
                [x1, y1, 0],
                [x1, y1, max_z + 10],
            ]
        )
        color = np.array(
            [
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [0, 1, 0, 1],
                [0, 1, 0, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ]
        )
        LineVisual.__init__(
            self, pos=verts, color=color, connect="segments", method="gl", **kwargs
        )


def vispy_process(queue, world_ltrb):
    def data_thread(queue, markers):
        while True:
            data = queue.get()
            if data is None:
                time.sleep(0.1)
                canvas.close()
                break
            points, face_colors, edge_colors = data
            if face_colors is not None and len(points) > 0:
                edge_width = 1 if isinstance(edge_colors, str) else 3
                markers.set_data(
                    points,
                    edge_width=edge_width,
                    edge_color=edge_colors,
                    face_color=face_colors,
                )
            else:
                markers.set_data(points)

    canvas = scene.SceneCanvas(
        title="3D Poses", bgcolor="w", size=(400, 400), show=True
    )
    view = canvas.central_widget.add_view()
    view.camera = cameras.TurntableCamera()
    view.camera.fov = 80
    view.camera.distance = (
        max(world_ltrb[2] - world_ltrb[0], world_ltrb[3] - world_ltrb[1]) * 0.8
    )  # 200 or 800
    visuals.create_visual_node(XYZAxisVisual)(world_ltrb, 100, parent=view.scene)
    x1, y1, x2, y2 = world_ltrb
    visuals.SurfacePlot(
        x=np.array([x1, x2]),
        y=np.array([y1, y2]),
        z=np.zeros((2, 2)),
        color=(0.5, 0.5, 0.5, 1),
        parent=view.scene,
    )

    markers = scene.visuals.Markers()
    markers.parent = view.scene

    # events
    def on_close(event):
        if event.text.lower() == "q":
            queue.put(None)

    canvas.events.key_press.connect(on_close)

    thread = Thread(target=data_thread, args=[queue, markers])
    thread.start()

    canvas.app.run()
    thread.join()


class Visualizer3D(object):
    def __init__(self, world_ltrb, color_generator):
        self.color_generator = color_generator
        self.world_ltrb = copy.deepcopy(world_ltrb)
        self.ori_wcx = np.mean(self.world_ltrb[0::2])
        self.ori_wcy = np.mean(self.world_ltrb[1::2])
        self.world_ltrb[0::2] -= self.ori_wcx
        self.world_ltrb[1::2] -= self.ori_wcy

        self.queue = Queue(maxsize=1)

        self.plot_process = Process(
            target=vispy_process, args=[self.queue, self.world_ltrb]
        )
        self.plot_process.start()

    def start(self):
        self.plot_process.start()

    def stop(self):
        try:
            self.queue.put(None, timeout=0.1)
        except Full:
            pass

    def update(self, data_3d):
        if data_3d is None:
            poses = []
            points = np.empty([0, 3], dtype=float)
        else:
            poses = data_3d["poses"]
            points = np.asarray([pose["points_3d"] for pose in poses]).reshape(-1, 3)
        face_colors = None
        edge_colors = "black"

        if len(poses) > 0:
            points[:, 0] -= self.ori_wcx
            points[:, 1] -= self.ori_wcy

            xmin, ymin, xmax, ymax = self.world_ltrb
            scores = np.asarray([pose["scores"] for pose in poses]).reshape(-1)
            keep = (
                (scores > 0.0)
                & (points[:, 0] > xmin)
                & (points[:, 0] < xmax)
                & (points[:, 1] > ymin)
                & (points[:, 1] < ymax)
            )
            points = points[keep]

            repeats = len(poses[0]["points_3d"])
            face_colors = self.get_colors([pose["id"] for pose in poses], repeats)
            face_colors = face_colors[keep]
        try:
            self.queue.put((points, face_colors, edge_colors), timeout=1)
            return True
        except Full:
            return False

    def get_colors(self, target_ids, repeats):
        colors = np.asarray(
            [self.color_generator.get_color(target_id) for target_id in target_ids],
            dtype=float,
        )
        colors /= 255.0
        colors = colors[:, ::-1]
        colors = np.repeat(colors, repeats, axis=0)
        return colors
