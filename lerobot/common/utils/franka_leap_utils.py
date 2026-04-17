import numpy as np

if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "object"):
    np.object = object
if not hasattr(np, "complex"):
    np.complex = complex
from urdfpy import URDF
import trimesh


class URDFHandPointCloud:
    def __init__(
        self,
        urdf_path="../../robot_descriptions/panda/panda_leap.urdf",
        hand_link_names=None,
        hand_link_keywords=("fingertip", "pip", "dip", "mcp", "palm"),
        use_collision=False,
        total_points=5000,
        sample_even=False,
    ):
        """
        Load URDF once, cache hand meshes once, pre-sample local points once.

        Args:
            urdf_path: path to URDF
            joint_names: ordered joint names corresponding to runtime qpos
            hand_link_names: explicit list of hand link names to include
            hand_link_keywords: fallback substring match on link names
            use_collision: use collision meshes instead of visual meshes
            total_points: total sampled points across all hand meshes
            sample_even: use sample_surface_even if True
        """
        self.robot = URDF.load(urdf_path)
        joint_names = [
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
            "panda_joint5", "panda_joint6", "panda_joint7",
            "leap_0", "leap_1", "leap_2", "leap_3", "leap_4",
            "leap_5", "leap_6", "leap_7", "leap_8", "leap_9", "leap_10", "leap_11", "leap_12",
            "leap_13", "leap_14", "leap_15"
        ]
        self.joint_names = list(joint_names)
        self.use_collision = use_collision
        self.total_points = int(total_points)
        self.sample_even = sample_even

        if hand_link_names is None and hand_link_keywords is None:
            raise ValueError("Provide hand_link_names or hand_link_keywords")

        self.hand_link_names = set(hand_link_names) if hand_link_names is not None else None
        self.hand_link_keywords = tuple(k.lower() for k in hand_link_keywords) if hand_link_keywords is not None else None

        # Each entry:
        # {
        #   "link": urdfpy.Link,
        #   "link_name": str,
        #   "T_link_mesh": (4,4),
        #   "local_points": (N,3),
        # }
        self.entries = []

        self._build_cache()

    def _is_hand_link(self, link_name):
        if self.hand_link_names is not None:
            return link_name in self.hand_link_names
        lname = link_name.lower()
        return any(k in lname for k in self.hand_link_keywords)

    def _sample_mesh_points(self, mesh, n_points):
        if n_points <= 0:
            return np.zeros((0, 3), dtype=np.float64)

        if self.sample_even:
            pts, _ = trimesh.sample.sample_surface_even(mesh, n_points)
        else:
            pts, _ = trimesh.sample.sample_surface(mesh, n_points)
        return np.asarray(pts, dtype=np.float64)

    def _build_cache(self):
        raw_mesh_entries = []

        for link in self.robot.links:
            if not self._is_hand_link(link.name):
                continue

            elems = link.collisions if self.use_collision else link.visuals

            for elem in elems:
                T_link_mesh = elem.origin if elem.origin is not None else np.eye(4)

                geom = elem.geometry
                if geom is None or geom.mesh is None:
                    # skip primitive boxes/cylinders/spheres in this version
                    continue

                mesh_container = geom.mesh
                meshes = mesh_container.meshes if hasattr(mesh_container, "meshes") else []
                if not meshes:
                    continue

                for mesh in meshes:
                    raw_mesh_entries.append({
                        "link": link,
                        "link_name": link.name,
                        "T_link_mesh": T_link_mesh.copy(),
                        "mesh": mesh.copy(),
                        "area": max(float(mesh.area), 1e-12),
                    })

        if not raw_mesh_entries:
            raise ValueError("No hand meshes found. Check hand link names/keywords and URDF mesh loading.")

        areas = np.array([e["area"] for e in raw_mesh_entries], dtype=np.float64)
        weights = areas / areas.sum()

        counts = np.floor(weights * self.total_points).astype(int)
        counts = np.maximum(counts, 1)

        # fix rounding so total is close to requested total_points
        diff = int(self.total_points - counts.sum())
        if diff > 0:
            order = np.argsort(-weights)
            for i in range(diff):
                counts[order[i % len(order)]] += 1
        elif diff < 0:
            order = np.argsort(weights)  # subtract from smallest first if possible
            remaining = -diff
            for idx in order:
                if remaining == 0:
                    break
                removable = min(remaining, max(0, counts[idx] - 1))
                counts[idx] -= removable
                remaining -= removable

        for e, n in zip(raw_mesh_entries, counts):
            local_points = self._sample_mesh_points(e["mesh"], int(n))
            self.entries.append({
                "link": e["link"],
                "link_name": e["link_name"],
                "T_link_mesh": e["T_link_mesh"],
                "local_points": local_points,
            })

    def qpos_to_cfg(self, qpos):
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if len(qpos) != len(self.joint_names):
            raise ValueError(f"Expected qpos of length {len(self.joint_names)}, got {len(qpos)}")
        return {name: float(val) for name, val in zip(self.joint_names, qpos)}

    @staticmethod
    def transform_points(T, points):
        """
        T: (4,4)
        points: (N,3)
        """
        R = T[:3, :3]
        t = T[:3, 3]
        return points @ R.T + t

    def get_hand_skeleton(self, qpos):
        cfg = self.qpos_to_cfg(qpos)
        link_fk = self.robot.link_fk(cfg=cfg)

        skeleton = []
        for e in self.entries:
            if e['link_name'] == "thumb_pip":
                continue
            T_world_link = link_fk[e["link"]]
            skeleton.append(T_world_link[:3, 3])
        skeleton = np.stack(skeleton, axis=0)
        return skeleton
    
    def get_point_cloud(self, qpos, base_transform=None):
        """
        Returns concatenated hand point cloud in world frame.

        Args:
            qpos: ordered joint positions matching self.joint_names
            base_transform: optional extra (4,4) transform applied on the left
                            e.g. camera_to_world or robot_base_to_world

        Returns:
            points_world: (N,3)
        """
        cfg = self.qpos_to_cfg(qpos)
        link_fk = self.robot.link_fk(cfg=cfg)

        pcs = []
        for e in self.entries:
            T_world_link = link_fk[e["link"]]
            T_world_mesh = T_world_link @ e["T_link_mesh"]

            if base_transform is not None:
                T_world_mesh = base_transform @ T_world_mesh

            pts_world = self.transform_points(T_world_mesh, e["local_points"])
            pcs.append(pts_world)

        if not pcs:
            return np.zeros((0, 3), dtype=np.float64)

        return np.concatenate(pcs, axis=0)
    

if __name__ == "__main__":
    joint_names = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
        "leap_0", "leap_1", "leap_2", "leap_3", "leap_4",
        "leap_5", "leap_6", "leap_7", "leap_8", "leap_9", "leap_10", "leap_11", "leap_12",
        "leap_13", "leap_14", "leap_15"
    ]

    hand_pc_model = URDFHandPointCloud(
        urdf_path="/home/yingyuan/lerobot/robot_descriptions/panda/panda_leap.urdf",
        joint_names=joint_names,
        hand_link_keywords=("fingertip", "pip", "dip", "mcp", "palm"),
        use_collision=False,
        total_points=1024,
    )

    qpos = np.array([ 1.7368, -0.3240, -1.5934, -1.7005, -0.2500,  1.9767, -0.2262, -0.0966,
        -0.1197, -0.1519,  0.7010, -0.0199, -0.0460, -0.1641,  0.7885, -0.0031,
        -0.0460, -0.1135,  0.9480,  0.3022, -0.0445,  0.1856,  0.8053])
    pc = hand_pc_model.get_point_cloud(qpos)

    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pcd])
    print("Point cloud shape:", pc.shape)
