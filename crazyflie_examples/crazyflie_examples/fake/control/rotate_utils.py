import torch

def rpm2angvel(rpms: torch.Tensor) -> torch.Tensor:
    """
    Convert rotor RPMs to angular velocities
    """
    return rpms * 2. * torch.pi / 60.

def angvel2rpm(ang_vel: torch.Tensor) -> torch.Tensor:
    """
    Convert angular velocities to rotor RPMs
    """
    return ang_vel * 60 / (2 * torch.pi)

def euler2quat(ang: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler angles (roll, pitch, yaw) to quaternions (qw, qx, qy, qz)
    """
    roll, pitch, yaw = ang.unbind(dim=-1)

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return torch.stack([qw, qx, qy, qz], dim=-1)

def quat2euler(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions (qw, qx, qy, qz) to Euler angles (roll, pitch, yaw)
    """
    qw, qx, qy, qz = quat.unbind(dim=-1)

    roll = torch.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    pitch = torch.asin(torch.clamp(2 * (qw * qy - qz * qx), -0.99999, 0.99999))
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

    return torch.stack([roll, pitch, yaw], dim=-1)

def euler2matrix(ang: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler angles (roll, pitch, yaw) to rotation matrix
    """
    roll, pitch, yaw = ang.unbind(dim=-1)

    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cr = torch.cos(roll)
    sr = torch.sin(roll)

    rot = torch.zeros(ang.shape[:-1] + (3, 3), dtype=ang.dtype, device=ang.device)
    rot[..., 0, 0] = cy * cp
    rot[..., 0, 1] = cy * sp * sr - sy * cr
    rot[..., 0, 2] = cy * sp * cr + sy * sr
    rot[..., 1, 0] = sy * cp
    rot[..., 1, 1] = sy * sp * sr + cy * cr
    rot[..., 1, 2] = sy * sp * cr - cy * sr
    rot[..., 2, 0] = -sp
    rot[..., 2, 1] = cp * sr
    rot[..., 2, 2] = cp * cr

    return rot

def quat2matrix(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions (qw, qx, qy, qz) to rotation matrix
    """
    qw, qx, qy, qz = quat.unbind(dim=-1)

    rot = torch.zeros(quat.shape[:-1] + (3, 3), dtype=quat.dtype, device=quat.device)
    rot[..., 0, 0] = 1 - 2 * qy**2 - 2 * qz**2
    rot[..., 0, 1] = 2 * (qx * qy - qz * qw)
    rot[..., 0, 2] = 2 * (qx * qz + qy * qw)
    rot[..., 1, 0] = 2 * (qx * qy + qz * qw)
    rot[..., 1, 1] = 1 - 2 * qx**2 - 2 * qz**2
    rot[..., 1, 2] = 2 * (qy * qz - qx * qw)
    rot[..., 2, 0] = 2 * (qx * qz - qy * qw)
    rot[..., 2, 1] = 2 * (qy * qz + qx * qw)
    rot[..., 2, 2] = 1 - 2 * qx**2 - 2 * qy**2

    return rot

def quat_distance(quat1: torch.Tensor, quat2: torch.Tensor) -> float:
    """
    Compute the distance between two quaternions (qw, qx, qy, qz)
    """
    assert quat1.shape == quat2.shape

    quat1 = quat1 / torch.norm(quat1, dim=-1, keepdim=True)
    quat2 = quat2 / torch.norm(quat2, dim=-1, keepdim=True)
    
    # make sure the dot product is in [-1, 1]
    return torch.acos(torch.clamp(torch.sum(quat1 * quat2, dim=-1), -1.0 + 1e-7, 1.0 - 1e-7))

def euler_distance(quat1: torch.Tensor, quat2: torch.Tensor) -> float:
    """
    Compute the mean of errors in roll, pitch, and yaw between two quaternions
    """
    euler1 = quat2euler(quat1)
    euler2 = quat2euler(quat2)

    # Adjust differences for 2π periodicity
    euler_diff = torch.abs((euler1 - euler2 + torch.pi) % (2 * torch.pi) - torch.pi)

    return euler_diff.mean(-1)

def inv_quat(quat: torch.Tensor) -> torch.Tensor:
    """
    Inverse a quaternion (qw, qx, qy, qz)
    """
    assert quat.shape[-1] == 4

    quat_inv = torch.zeros_like(quat)
    quat_inv[..., 0] = quat[..., 0]
    quat_inv[..., 1:] = -quat[..., 1:]
    quat_inv = quat_inv / torch.norm(quat_inv, dim=-1, keepdim=True)

    return quat_inv

def delta_quat(quat1: torch.Tensor, quat2: torch.Tensor) -> torch.Tensor:
    """
    Compute the delta quaternion from quat1 to quat2
    """
    assert quat1.shape == quat2.shape

    quat1 = quat1 / torch.norm(quat1, dim=-1, keepdim=True)
    quat2 = quat2 / torch.norm(quat2, dim=-1, keepdim=True)

    delta_quat = quat_multiply(quat2, inv_quat(quat1))
    delta_quat = delta_quat / torch.norm(delta_quat, dim=-1, keepdim=True)

    return delta_quat

def quat_multiply(quat1: torch.Tensor, quat2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions (qw, qx, qy, qz)
    """
    assert quat1.shape == quat2.shape

    qw1, qx1, qy1, qz1 = quat1.unbind(dim=-1)
    qw2, qx2, qy2, qz2 = quat2.unbind(dim=-1)

    quat = torch.zeros(quat1.shape[:-1] + (4,), dtype=quat1.dtype, device=quat1.device)
    quat[..., 0] = qw1 * qw2 - qx1 * qx2 - qy1 * qy2 - qz1 * qz2
    quat[..., 1] = qw1 * qx2 + qx1 * qw2 + qy1 * qz2 - qz1 * qy2
    quat[..., 2] = qw1 * qy2 - qx1 * qz2 + qy1 * qw2 + qz1 * qx2
    quat[..., 3] = qw1 * qz2 + qx1 * qy2 - qy1 * qx2 + qz1 * qw2

    return quat

def quat_axis(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert a quaternion (qw, qx, qy, qz) to axis-angle representation (angle, x, y, z)
    """
    qw, qx, qy, qz = quat.unbind(dim=-1)

    angle = 2 * torch.acos(qw)
    x = qx / torch.sqrt(1 - qw**2)
    y = qy / torch.sqrt(1 - qw**2)
    z = qz / torch.sqrt(1 - qw**2)

    return torch.stack([angle, x, y, z], dim=-1)

def rotate_vector(vec: torch.Tensor, rot: torch.Tensor, mode: str = 'matrix') -> torch.Tensor:
    """
    Rotate a vector by a rotation matrix or quaternion or Euler angles, depending on the mode ('matrix', 'quat', 'euler')
    With rot representing the orienation of a rigid body, this func is to transform the vec from the body frame to the world frame.
    """
    assert vec.shape[-1] == 3

    if mode == 'matrix':
        assert rot.shape[-2:] == (3, 3) and rot.shape[:-2] == vec.shape[:-1]
        return torch.bmm(rot, vec.unsqueeze(-1)).squeeze(-1)
    elif mode == 'euler':
        assert rot.shape[-1] == 3 and rot.shape[:-1] == vec.shape[:-1]
        return torch.bmm(euler2matrix(rot), vec.unsqueeze(-1)).squeeze(-1)
    elif mode == 'quat':
        assert rot.shape[-1] == 4 and rot.shape[:-1] == vec.shape[:-1]
        qw, qvec = rot[..., :1], rot[..., 1:]
        a = vec * (2.0 * qw ** 2 - 1.0)
        b = torch.cross(qvec, vec, dim=-1) * (2.0 * qw)
        c = qvec * torch.sum(qvec * vec, dim=-1, keepdim=True) * 2.0
        return a + b + c
    else:
        raise ValueError('Invalid mode: {}'.format(mode))
    
def inv_rotate_vector(vec: torch.Tensor, rot: torch.Tensor, mode: str = 'matrix') -> torch.Tensor:
    """
    Inverse rotate a vector by a rotation matrix or quaternion or Euler angles, depending on the mode ('matrix', 'quat', 'euler')
    With rot representing the orienation of a rigid body, this func is to transform the vec from the world frame to the body frame.
    """
    assert vec.shape[-1] == 3
    
    if mode == 'matrix':
        assert rot.shape[-2:] == (3, 3) and rot.shape[:-2] == vec.shape[:-1]
        return torch.bmm(rot.transpose(-2, -1), vec.unsqueeze(-1)).squeeze(-1)
    elif mode == 'euler':
        assert rot.shape[-1] == 3 and rot.shape[:-1] == vec.shape[:-1]
        return torch.bmm(euler2matrix(rot).transpose(-2, -1), vec.unsqueeze(-1)).squeeze(-1)
    elif mode == 'quat':
        assert rot.shape[-1] == 4 and rot.shape[:-1] == vec.shape[:-1]
        qw, qvec = rot[..., :1], rot[..., 1:]
        a = vec * (2.0 * qw ** 2 - 1.0)
        b = torch.cross(vec, qvec, dim=-1) * (2.0 * qw)
        c = qvec * torch.sum(qvec * vec, dim=-1, keepdim=True) * 2.0
        return a + b + c
    else:
        raise ValueError('Invalid mode: {}'.format(mode))
    
def rotate_axis(axis: int, rot: torch.Tensor, mode: str = 'euler') -> torch.Tensor:
    """
    Get axis direction after rotation by a rotation matrix or quaternion or Euler angles, depending on the mode ('matrix', 'quat', 'euler')
    """
    if mode == 'matrix':
        basis_vec = torch.zeros(rot.shape[:-2] + (3,), dtype=rot.dtype, device=rot.device)
    else:
        basis_vec = torch.zeros(rot.shape[:-1] + (3,), dtype=rot.dtype, device=rot.device)

    basis_vec[..., axis] = 1.0
    return rotate_vector(basis_vec, rot, mode)
    
def quat_left(quat: torch.Tensor) -> torch.Tensor:
    """
    Compute the left matrix of a quaternion (qw, qx, qy, qz)
    """
    qw, qx, qy, qz = quat.unbind(dim=-1)

    left_matrix = torch.stack([
        torch.stack([qw, -qx, -qy, -qz], dim=-1),
        torch.stack([qx, qw, -qz, qy], dim=-1),
        torch.stack([qy, qz, qw, -qx], dim=-1),
        torch.stack([qz, -qy, qx, qw], dim=-1)
    ], dim=-2)

    return left_matrix

def quat_right(quat: torch.Tensor) -> torch.Tensor:
    """
    Compute the right matrix of a quaternion (qw, qx, qy, qz)
    """
    qw, qx, qy, qz = quat.unbind(dim=-1)

    right_matrix = torch.stack([
        torch.stack([qw, -qx, -qy, -qz], dim=-1),
        torch.stack([qx, qw, qz, -qy], dim=-1),
        torch.stack([qy, -qz, qw, qx], dim=-1),
        torch.stack([qz, qy, -qx, qw], dim=-1)
    ], dim=-2)

    return right_matrix

def omega_rotate_from_euler(vec: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    """
    Convert Angular rates from world frame to body frame using Euler angle(roll, pitch, yaw)
    """
    if ang.ndim == 1:
        ang = ang.expand_as(vec)
    
    roll,pitch, yaw = ang.unbind(dim=-1)

    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cr = torch.cos(roll)
    sr = torch.sin(roll)

    rot = torch.zeros(ang.shape[:-1] + (3, 3), dtype=ang.dtype, device=ang.device)
    rot[..., 0, 0] = torch.ones_like(cy)
    rot[..., 0, 1] = torch.zeros_like(cy)
    rot[..., 0, 2] = -sp
    rot[..., 1, 0] = torch.zeros_like(cy)
    rot[..., 1, 1] = cr
    rot[..., 1, 2] = cp * sr
    rot[..., 2, 0] = torch.zeros_like(cy)
    rot[..., 2, 1] = -sr
    rot[..., 2, 2] = cp * cr

    assert rot.shape[:-2] == vec.shape[:-1]
    return torch.bmm(rot, vec.unsqueeze(-1)).squeeze(-1)


if __name__ == "__main__":
    from scipy.spatial.transform import Rotation as R
    
    ang = torch.rand(3, 3) * 360. - 180.
    # vec = torch.tensor([[1., 0., 0.]])
    vec = torch.rand(3, 3)
    
    print('ang:', ang)
    print('vec:', vec)
    
    rot_vec_1 = rotate_vector(vec, ang, mode='euler')
    rot_vec_2 = rotate_vector(vec, euler2quat(ang), mode='quat')
    rot_vec_3 = rotate_vector(vec, euler2matrix(ang), mode='matrix')
    
    inv_rot_vec_1 = inv_rotate_vector(rot_vec_1, ang, mode='euler')
    inv_rot_vec_2 = inv_rotate_vector(rot_vec_2, euler2quat(ang), mode='quat')
    inv_rot_vec_3 = inv_rotate_vector(rot_vec_3, euler2matrix(ang), mode='matrix')
    
    print(rot_vec_1)
    print(rot_vec_2)
    print(rot_vec_3)
    
    print(inv_rot_vec_1)
    print(inv_rot_vec_2)
    print(inv_rot_vec_3)