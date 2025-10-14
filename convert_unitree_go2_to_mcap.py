from urdfpy import URDF
import argparse
import pyrender
import matplotlib.pyplot as plt
import numpy as np

URDF_FILE = "urdf/go2_description.urdf"

# def look_at(camera_pos, target, up=np.array([0,0,1])):
#     camera_pos = np.array(camera_pos, dtype=float)
#     target = np.array(target, dtype=float)
#     up = np.array(up, dtype=float)

#     # Forward vector (camera looks along -Z)
#     forward = (camera_pos - target)
#     forward /= np.linalg.norm(forward)

#     # Right vector
#     right = np.cross(up, forward)
#     right /= np.linalg.norm(right)

#     # True up vector
#     up_corrected = np.cross(forward, right)

#     # Build rotation-translation matrix
#     pose = np.eye(4)
#     pose[0, :3] = right
#     pose[1, :3] = up_corrected
#     pose[2, :3] = forward
#     pose[:3, 3] = camera_pos
#     return pose


if __name__ == "__main__":


    argparser = argparse.ArgumentParser()
    argparser.add_argument("--urdf", type=str, default=URDF_FILE)
    args = argparser.parse_args()

    print(f"Loading URDF from {args.urdf} ...")
    robot = URDF.load(args.urdf)

    joint_positions = {}
    
    for joint in robot.actuated_joints:
        print(joint.name)
        joint_positions[joint.name] = 0.0


    robot.show(cfg=joint_positions)
    # fk_poses = robot.link_fk(cfg=joint_positions,
    #                          links=['panda_link0', 
    #                                 'panda_link1',
    #                                 'panda_link2', 
    #                                 'panda_link3', 
    #                                 'panda_link4',
    #                                 'panda_link5',
    #                                 'panda_link6',
    #                                 'panda_link7',
    #                                 'panda_link8'])

    # for key, value in fk_poses.items():
    #     print(key.name)
    #     print(value)


    # # robot.show()

    # scene = pyrender.Scene()
    # meshes = robot.visual_trimesh_fk()
    # for tm, transform in meshes.items():
    #     pm = pyrender.Mesh.from_trimesh(tm)
    #     scene.add(pm, pose=transform)

    # # Add a camera
    # camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)


    # pose = look_at([0, 1, 1.5], [0, 0, 0], [0, 0, 1])

    # scene.add(camera, pose=pose)

    # # Add some light (optional, but pyrender renderer wants light sources)
    # light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    # scene.add(light, pose=np.eye(4))

    # # Render
    # renderer = pyrender.OffscreenRenderer(512, 512)
    # color, depth = renderer.render(scene)

    # mask = depth > 0

    # plt.imshow(mask, cmap='gray')
    # plt.show()