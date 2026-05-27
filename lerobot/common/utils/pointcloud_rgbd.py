import torch
import numpy as np
from PIL import Image

import plotly.graph_objects as go
import numpy as np
import torch

def visualize_pointcloud_plotly(points, colors, max_points=50000, save_path="pointcloud.html"):
    """
    Visualize point cloud with colors using Plotly.
    
    Args:
        points: (N, 3) numpy array or torch tensor of point coordinates
        colors: (N, 3) numpy array or torch tensor of RGB colors (values in 0-1 or 0-255)
        max_points: Maximum number of points to display (for performance)
        save_path: Path to save the HTML file
    """
    # Convert to numpy if torch tensor
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    if torch.is_tensor(colors):
        colors = colors.cpu().numpy()
    
    # Ensure colors are in 0-1 range for plotly
    if colors.max() > 1.0:
        colors = colors / 255.0
    
    # Sample points if too many
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        colors = colors[indices]
    
    # Create RGB colors as hex strings for plotly
    rgb_colors = [f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})' 
                  for r, g, b in colors]
    
    # Create 3D scatter plot
    fig = go.Figure(data=[
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=rgb_colors,
                opacity=0.8
            ),
            hovertemplate='X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<br>RGB: %{text}<extra></extra>',
            text=[f'({r*255:.0f}, {g*255:.0f}, {b*255:.0f})' for r, g, b in colors]
        )
    ])
    
    # Update layout
    fig.update_layout(
        title="Point Cloud Visualization",
        scene=dict(
            xaxis_title="X (World)",
            yaxis_title="Y (World)",
            zaxis_title="Z (World)",
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1000,
        height=800
    )
    
    # Save as HTML
    fig.write_html(save_path)
    print(f"Saved point cloud visualization to {save_path}")
    
    return fig

def create_structured_cylinder(center_x=0.1, center_y=0.1, radius=0.02, height=0.04, 
                                num_points_circumference=200, num_points_height=50):
    """
    Create a properly structured cylinder with evenly distributed points.
    Not random scattering - structured grid!
    
    Args:
        center_x, center_y: Center position
        radius: Cylinder radius
        height: Cylinder height
        num_points_circumference: Number of points around the circle
        num_points_height: Number of points along height
    """
    points_list = []
    colors_list = []
    
    # 1. Side surface - structured grid
    theta = torch.linspace(0, 2*np.pi, num_points_circumference)
    z = torch.linspace(0, height, num_points_height)
    
    # Create meshgrid for structured points
    theta_grid, z_grid = torch.meshgrid(theta, z, indexing='ij')
    
    # Calculate points on side surface
    x_side = center_x + radius * torch.cos(theta_grid)
    y_side = center_y + radius * torch.sin(theta_grid)
    z_side = z_grid
    
    # Colors for side: Red varies with x, Blue varies with y
    red_side = (x_side - (center_x - radius)) / (2 * radius)
    blue_side = (y_side - (center_y - radius)) / (2 * radius)
    green_side = torch.ones_like(red_side) * 0.5
    
    points_side = torch.stack([x_side.flatten(), y_side.flatten(), z_side.flatten()], dim=1)
    colors_side = torch.stack([red_side.flatten(), green_side.flatten(), blue_side.flatten()], dim=1)
    
    points_list.append(points_side)
    colors_list.append(colors_side)
    
    # 2. Top cap - structured grid
    r_top = torch.linspace(0, radius, int(np.sqrt(num_points_circumference * num_points_height / 2)))
    theta_top = torch.linspace(0, 2*np.pi, int(2 * len(r_top)))
    
    r_grid, theta_grid = torch.meshgrid(r_top, theta_top, indexing='ij')
    x_top = center_x + r_grid * torch.cos(theta_grid)
    y_top = center_y + r_grid * torch.sin(theta_grid)
    z_top = torch.ones_like(x_top) * height
    
    red_top = (x_top - (center_x - radius)) / (2 * radius)
    blue_top = (y_top - (center_y - radius)) / (2 * radius)
    green_top = torch.ones_like(red_top) * 0.5
    
    points_top = torch.stack([x_top.flatten(), y_top.flatten(), z_top.flatten()], dim=1)
    colors_top = torch.stack([red_top.flatten(), green_top.flatten(), blue_top.flatten()], dim=1)
    
    points_list.append(points_top)
    colors_list.append(colors_top)
    
    # 3. Bottom cap - structured grid
    x_bottom = center_x + r_grid * torch.cos(theta_grid)
    y_bottom = center_y + r_grid * torch.sin(theta_grid)
    z_bottom = torch.zeros_like(x_bottom)
    
    red_bottom = (x_bottom - (center_x - radius)) / (2 * radius)
    blue_bottom = (y_bottom - (center_y - radius)) / (2 * radius)
    green_bottom = torch.ones_like(red_bottom) * 0.5
    
    points_bottom = torch.stack([x_bottom.flatten(), y_bottom.flatten(), z_bottom.flatten()], dim=1)
    colors_bottom = torch.stack([red_bottom.flatten(), green_bottom.flatten(), blue_bottom.flatten()], dim=1)
    
    points_list.append(points_bottom)
    colors_list.append(colors_bottom)
    
    # Concatenate all
    points = torch.cat(points_list, dim=0)
    colors = torch.cat(colors_list, dim=0)
    
    return points, colors

def create_socket_with_structured_points():
    """
    Create a socket (small cylinder) with properly structured points.
    Random offset for placement.
    """
    # Random offset between -0.02 and 0.02
    random_offset_x = (torch.rand(1) - 0.5) * 0.04
    random_offset_y = (torch.rand(1) - 0.5) * 0.04
    
    # Center position (around 0.1, 0.1)
    center_x = 0.1 + random_offset_x.item()
    center_y = 0.1 + random_offset_y.item()
    
    print(f"Socket center: x={center_x:.4f}, y={center_y:.4f}")
    
    # Create structured cylinder
    points, colors = create_structured_cylinder(
        center_x=center_x,
        center_y=center_y,
        radius=0.02,
        height=0.04,
        num_points_circumference=100,  # 100 points around circle
        num_points_height=30            # 30 points along height
    )
    
    return points, colors, center_x, center_y

def render_top_down_custom(
    points,
    colors,
    center_x=0.1,
    center_y=0.1,
    H=480,
    W=640,
    fov_deg=60,
    camera_height_offset=0.02,
    brightness_scale=1.0,
    point_radius=2,
    background_color=(0.0, 0.0, 0.0),
):
    """
    Render a point cloud from a top-down camera using point splatting.

    Camera convention:
    - camera looks toward world -Z
    - image up is world -X
    - image right is world +Y

    Args:
        points: (N, 3) torch tensor
        colors: (N, 3) torch tensor, either in [0,1] or [0,255]
        H, W: output image size
        fov_deg: vertical field of view
        camera_height_offset: camera height above top of object
        brightness_scale: multiply colors before clamping
        point_radius: splat radius in pixels
        background_color: tuple of 3 floats in [0,1]

    Returns:
        rgb_img: (H, W, 3) torch tensor in [0,1]
        depth_img: (H, W) torch tensor
    """
    device = points.device
    points = points.float()
    colors = colors.float()
    # Normalize colors if needed
    if colors.max() > 1.0:
        colors = colors / 255.0

    # Brightness adjustment
    colors = torch.clamp(colors * brightness_scale, 0.0, 1.0)
    # # Estimate object center and top
    top_z = points[:, 2].max()

    camera_pos = torch.tensor(
        [center_x, center_y, top_z + camera_height_offset],
        device=device,
        dtype=torch.float32,
    )
    camera_target = torch.tensor(
        [center_x, center_y, top_z],
        device=device,
        dtype=torch.float32,
    )

    def look_at(eye, target, up):
        forward = target - eye
        forward = forward / torch.norm(forward)

        right = torch.cross(forward, up, dim=0)
        right = right / torch.norm(right)

        up_cam = torch.cross(right, forward, dim=0)
        up_cam = up_cam / torch.norm(up_cam)

        view = torch.eye(4, device=device, dtype=torch.float32)
        view[0, :3] = right
        view[1, :3] = up_cam
        view[2, :3] = -forward
        view[0, 3] = -torch.dot(right, eye)
        view[1, 3] = -torch.dot(up_cam, eye)
        view[2, 3] = torch.dot(forward, eye)
        return view

    # Image up = world -X
    world_up = torch.tensor([-1.0, 0.0, 0.0], device=device, dtype=torch.float32)
    view_mat = look_at(camera_pos, camera_target, world_up)

    # Perspective projection
    aspect = W / H
    fov_rad = torch.tensor(fov_deg * np.pi / 180.0, device=device, dtype=torch.float32)
    f = 1.0 / torch.tan(fov_rad / 2.0)
    near = 0.001
    far = float(camera_height_offset + 0.1)

    proj_mat = torch.zeros((4, 4), device=device, dtype=torch.float32)
    proj_mat[0, 0] = f / aspect
    proj_mat[1, 1] = f
    proj_mat[2, 2] = (far + near) / (near - far)
    proj_mat[2, 3] = (2 * far * near) / (near - far)
    proj_mat[3, 2] = -1.0

    # Homogeneous coordinates
    ones = torch.ones((points.shape[0], 1), device=device, dtype=torch.float32)
    points_h = torch.cat([points, ones], dim=1)

    # World -> view -> clip
    points_view = points_h @ view_mat.T
    points_clip = points_view @ proj_mat.T

    # Keep points in front of camera
    valid = points_view[:, 2] < 0
    if not valid.any():
        print("No points in front of camera.")
        return None, None

    points_view = points_view[valid]
    points_clip = points_clip[valid]
    colors = colors[valid]

    # Perspective divide
    w = points_clip[:, 3]
    eps = 1e-8
    ndc = points_clip[:, :3] / (w[:, None] + eps)

    # NDC -> image coords
    u = (ndc[:, 0] + 1.0) * 0.5 * W
    v = (1.0 - (ndc[:, 1] + 1.0) * 0.5) * H
    z = -points_view[:, 2]

    # Filter valid image points
    in_bounds = (
        (u >= 0) & (u < W) &
        (v >= 0) & (v < H) &
        (z > near) & (z < far)
    )

    u = u[in_bounds]
    v = v[in_bounds]
    z = z[in_bounds]
    colors = colors[in_bounds]

    if len(u) == 0:
        print("No points projected into the image.")
        return None, None

    # Sort from near to far so closer points overwrite farther ones
    order = torch.argsort(z)
    u = u[order]
    v = v[order]
    z = z[order]
    colors = colors[order]

    # Initialize images
    bg = torch.tensor(background_color, device=device, dtype=torch.float32)
    rgb_img = bg.view(1, 1, 3).repeat(H, W, 1).clone()
    depth_img = torch.full((H, W), far, device=device, dtype=torch.float32)

    # Point splatting
    for i in range(len(u)):
        ui = int(round(u[i].item()))
        vi = int(round(v[i].item()))
        zi = z[i]
        ci = colors[i]

        u0 = max(0, ui - point_radius)
        u1 = min(W - 1, ui + point_radius)
        v0 = max(0, vi - point_radius)
        v1 = min(H - 1, vi + point_radius)

        for vv in range(v0, v1 + 1):
            for uu in range(u0, u1 + 1):
                # circular splat
                if (uu - ui) ** 2 + (vv - vi) ** 2 > point_radius ** 2:
                    continue

                if zi < depth_img[vv, uu]:
                    depth_img[vv, uu] = zi
                    rgb_img[vv, uu] = ci

    filled = (depth_img < far).sum().item()

    if filled > 0:
        rendered_colors = rgb_img[depth_img < far]
        print(
            f"Rendered mean RGB: "
            f"({rendered_colors[:,0].mean():.3f}, "
            f"{rendered_colors[:,1].mean():.3f}, "
            f"{rendered_colors[:,2].mean():.3f})"
        )

    return rgb_img, depth_img

def render_bottom_up_custom(
    points,
    colors,
    center_x=0.1,
    center_y=0.1,
    H=480,
    W=640,
    fov_deg=60,
    camera_height_offset=0.02,
    brightness_scale=1.0,
    point_radius=2,
    background_color=(0.0, 0.0, 0.0),
):
    """
    Render a point cloud from a bottom-up camera using point splatting.

    Camera convention:
    - camera looks toward world +Z
    - image up is world +X
    - image right is world +Y
    """
    device = points.device
    points = points.float()
    colors = colors.float()
    if colors.max() > 1.0:
        colors = colors / 255.0

    colors = torch.clamp(colors * brightness_scale, 0.0, 1.0)
    bottom_z = points[:, 2].min()

    camera_pos = torch.tensor(
        [center_x, center_y, bottom_z - camera_height_offset],
        device=device,
        dtype=torch.float32,
    )
    camera_target = torch.tensor(
        [center_x, center_y, bottom_z],
        device=device,
        dtype=torch.float32,
    )

    def look_at(eye, target, up):
        forward = target - eye
        forward = forward / torch.norm(forward)

        right = torch.cross(forward, up, dim=0)
        right = right / torch.norm(right)

        up_cam = torch.cross(right, forward, dim=0)
        up_cam = up_cam / torch.norm(up_cam)

        view = torch.eye(4, device=device, dtype=torch.float32)
        view[0, :3] = right
        view[1, :3] = up_cam
        view[2, :3] = -forward
        view[0, 3] = -torch.dot(right, eye)
        view[1, 3] = -torch.dot(up_cam, eye)
        view[2, 3] = torch.dot(forward, eye)
        return view

    world_up = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)
    view_mat = look_at(camera_pos, camera_target, world_up)

    aspect = W / H
    fov_rad = torch.tensor(fov_deg * np.pi / 180.0, device=device, dtype=torch.float32)
    f = 1.0 / torch.tan(fov_rad / 2.0)
    near = 0.001
    far = float(camera_height_offset + 0.1)

    proj_mat = torch.zeros((4, 4), device=device, dtype=torch.float32)
    proj_mat[0, 0] = f / aspect
    proj_mat[1, 1] = f
    proj_mat[2, 2] = (far + near) / (near - far)
    proj_mat[2, 3] = (2 * far * near) / (near - far)
    proj_mat[3, 2] = -1.0

    ones = torch.ones((points.shape[0], 1), device=device, dtype=torch.float32)
    points_h = torch.cat([points, ones], dim=1)

    points_view = points_h @ view_mat.T
    points_clip = points_view @ proj_mat.T

    valid = points_view[:, 2] < 0
    if not valid.any():
        print("No points in front of camera.")
        return None, None

    points_view = points_view[valid]
    points_clip = points_clip[valid]
    colors = colors[valid]

    w = points_clip[:, 3]
    eps = 1e-8
    ndc = points_clip[:, :3] / (w[:, None] + eps)

    u = (ndc[:, 0] + 1.0) * 0.5 * W
    v = (1.0 - (ndc[:, 1] + 1.0) * 0.5) * H
    z = -points_view[:, 2]

    in_bounds = (
        (u >= 0) & (u < W) &
        (v >= 0) & (v < H) &
        (z > near) & (z < far)
    )

    u = u[in_bounds]
    v = v[in_bounds]
    z = z[in_bounds]
    colors = colors[in_bounds]

    if len(u) == 0:
        print("No points projected into the image.")
        return None, None

    order = torch.argsort(z)
    u = u[order]
    v = v[order]
    z = z[order]
    colors = colors[order]

    bg = torch.tensor(background_color, device=device, dtype=torch.float32)
    rgb_img = bg.view(1, 1, 3).repeat(H, W, 1).clone()
    depth_img = torch.full((H, W), far, device=device, dtype=torch.float32)

    for i in range(len(u)):
        ui = int(round(u[i].item()))
        vi = int(round(v[i].item()))
        zi = z[i]
        ci = colors[i]

        u0 = max(0, ui - point_radius)
        u1 = min(W - 1, ui + point_radius)
        v0 = max(0, vi - point_radius)
        v1 = min(H - 1, vi + point_radius)

        for vv in range(v0, v1 + 1):
            for uu in range(u0, u1 + 1):
                if (uu - ui) ** 2 + (vv - vi) ** 2 > point_radius ** 2:
                    continue

                if zi < depth_img[vv, uu]:
                    depth_img[vv, uu] = zi
                    rgb_img[vv, uu] = ci

    filled = (depth_img < far).sum().item()

    if filled > 0:
        rendered_colors = rgb_img[depth_img < far]
        print(
            f"Rendered mean RGB: "
            f"({rendered_colors[:,0].mean():.3f}, "
            f"{rendered_colors[:,1].mean():.3f}, "
            f"{rendered_colors[:,2].mean():.3f})"
        )

    return rgb_img, depth_img

if __name__ == "__main__":


    # Run the experiment
    print("=" * 60)
    print("CREATING COLORED CYLINDER")
    print("=" * 60)

    # Create cylinder
    points, colors , _, _= create_socket_with_structured_points()
    # data=torch.load("./pointcloud.pth")
    # points= data["points"].cpu()
    # colors= data["colors"].cpu()
    # points = data["points"].cpu().float()
    # colors = data["colors"].cpu().float()

    if colors.max() > 1.0:
        colors = colors / 255.0
    fig = visualize_pointcloud_plotly(
            points, 
            colors, 
            max_points=50000,
            save_path="cylinder_pointcloud.html"
        )
    print(f"Points shape: {points.shape}")
    print(f"X range: [{points[:,0].min():.2f}, {points[:,0].max():.2f}]")
    print(f"Y range: [{points[:,1].min():.2f}, {points[:,1].max():.2f}]")
    print(f"Z range: [{points[:,2].min():.2f}, {points[:,2].max():.2f}]")
    print(f"Color range - Red: [{colors[:,0].min():.2f}, {colors[:,0].max():.2f}]")
    print(f"Color range - Blue: [{colors[:,2].min():.2f}, {colors[:,2].max():.2f}]")

    print("\n" + "=" * 60)
    print("RENDERING TOP-DOWN VIEW")
    print("Camera: up = -X, right = +Y, looking down")
    print("=" * 60)

    # Render
    # rgb_img, depth_img = render_top_down_custom(points, colors, H=480, W=960, camera_height_offset=0.08, fov_deg=30, brightness_scale=1, point_radius=5)

    # if rgb_img is not None:
    #     # Convert to numpy and save
    #     rgb_np = (rgb_img.cpu().numpy() * 255).astype(np.uint8)
        
    #     # Save RGB image
    #     Image.fromarray(rgb_np).save("cylinder_top_down.png")
    #     print("\n✓ Saved: cylinder_top_down.png")
        
    #     # Also save depth visualization
    #     depth_np = depth_img.cpu().numpy()
    #     depth_normalized = (depth_np / depth_np.max() * 255).astype(np.uint8)
    #     Image.fromarray(depth_normalized).save("cylinder_top_down_depth.png")
    #     print("✓ Saved: cylinder_top_down_depth.png")
        
    #     # Print some debug info
    #     print("\nDebug Info:")
    #     print(f"  Image size: {rgb_np.shape}")
    #     print(f"  Non-black pixels: {(rgb_np.sum(axis=2) > 0).sum()}")
        
    #     # Check colors at specific pixels (center, edges)
    #     center_u, center_v = 320, 240
    #     print(f"\n  Color at center (u={center_u}, v={center_v}): {rgb_np[center_v, center_u]}")
        
    #     # Expected: Center x≈1, y≈0 should have red≈0.5, blue≈0.5
    #     print("  Expected center: red~128, blue~128 (since x=1→red=0.5, y=0→blue=0.5)")
        
    # else:
    #     print("Failed to render!")

    # # Render
    rgb_img, depth_img = render_bottom_up_custom(points, colors, H=480, W=960, camera_height_offset=0.08, fov_deg=30, brightness_scale=1, point_radius=5)

    if rgb_img is not None:
        # Convert to numpy and save
        rgb_np = (rgb_img.cpu().numpy() * 255).astype(np.uint8)
        
        # Save RGB image
        Image.fromarray(rgb_np).save("cylinder_bottom_up.png")
        print("\n✓ Saved: cylinder_bottom_up.png")
        
        # Also save depth visualization
        depth_np = depth_img.cpu().numpy()
        depth_normalized = (depth_np / depth_np.max() * 255).astype(np.uint8)
        Image.fromarray(depth_normalized).save("cylinder_bottom_up_depth.png")
        print("✓ Saved: cylinder_bottom_up_depth.png")
        
        # Print some debug info
        print("\nDebug Info:")
        print(f"  Image size: {rgb_np.shape}")
        print(f"  Non-black pixels: {(rgb_np.sum(axis=2) > 0).sum()}")
        
        # Check colors at specific pixels (center, edges)
        center_u, center_v = 320, 240
        print(f"\n  Color at center (u={center_u}, v={center_v}): {rgb_np[center_v, center_u]}")
        
        # Expected: Center x≈1, y≈0 should have red≈0.5, blue≈0.5
        print("  Expected center: red~128, blue~128 (since x=1→red=0.5, y=0→blue=0.5)")
        
    else:
        print("Failed to render!")