import math
import tensorflow as tf

# from kornia.geometry.subpix import dsnt
# from kornia.utils.grid import create_meshgrid


class FineMatching(tf.keras.Model):
    """FineMatching with s2d paradigm"""

    def __init__(self):
        super().__init__()

    def call(self, feat_f0, feat_f1, data, training = False):
        """
        Args:
            feat0 (torch.Tensor): [M, WW, C]
            feat1 (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_f0.shape
        W = int(math.sqrt(WW))
        scale = data['hw0_i'][0] / data['hw0_f'][0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

        # corner case: if no coarse matches found
        if M == 0:
            assert training == False, "M is always >0, when training, see coarse_matching.py"
            # logger.warning('No matches found in coarse-level.')
            print('No matches found in coarse-level.')
            data.update({
                # 'expec_f': torch.empty(0, 3, device=feat_f0.device),
                'expec_f': tf.zeros([0, 3]),
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            return data

        feat_f0_picked = feat_f0_picked = feat_f0[:, WW//2, :]
        sim_matrix = tf.einsum('mc,mrc->mr', feat_f0_picked, feat_f1)
        softmax_temp = 1. / C**.5
        heatmap =  tf.reshape(tf.nn.softmax(softmax_temp * sim_matrix, axis=1),(-1, W, W))

        # compute coordinates from heatmap
        coords_normalized = self.spatial_expec(heatmap[None], True)[0]# dsnt.spatial_expectation2d(heatmap[None], True)[0]  # [M, 2]

        grid_normalized = self.create_mesh(W, W, True) # [1, WW, 2]
        grid_normalized = tf.reshape(grid_normalized,(1, -1, 2))

        # compute std over <x, y>
        var = tf.reduce_sum(tf.cast((grid_normalized**2),tf.double) * tf.cast(tf.reshape(heatmap,(-1, WW, 1)),tf.double), axis=1) - coords_normalized**2  # [M, 2]
        std = tf.reduce_sum(tf.math.sqrt(tf.clip_by_value(var, clip_value_min=1e-10,clip_value_max=float('inf'))), -1)  # [M]  clamp needed for numerical stability

        
        # for fine-level supervision
        data.update({'expec_f': tf.concat([coords_normalized,tf.expand_dims(std, axis=1)], axis=-1)})

        # compute absolute kpt coords
        data = self.get_fine_match(coords_normalized, data)
        print("module 5 Done")
        return data


    def get_fine_match(self, coords_normed, data):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale

        # mkpts0_f and mkpts1_f
        mkpts0_f = data['mkpts0_c']
        scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else scale
        mkpts1_f = data['mkpts1_c'] + (coords_normed * (W // 2) * scale1)[:len(data['mconf'])]

        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f
        })

        return data

    
    def create_mesh(self,height: int,width: int,normalized_coordinates: bool = True):

        xs = tf.linspace(0, width - 1, width)
        ys = tf.linspace(0, height - 1, height)
        # Fix TracerWarning
        # Note: normalize_pixel_coordinates still gots TracerWarning since new width and height
        #       tensors will be generated.
        # Below is the code using normalize_pixel_coordinates:
        # base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys]), dim=2)
        # if normalized_coordinates:
        #     base_grid = K.geometry.normalize_pixel_coordinates(base_grid, height, width)
        # return torch.unsqueeze(base_grid.transpose(0, 1), dim=0)
        if normalized_coordinates:
            xs = (xs / (width - 1) - 0.5) * 2
            ys = (ys / (height - 1) - 0.5) * 2
        # generate grid by stacking coordinates

        base_grid = tf.stack(tf.meshgrid(xs, ys, indexing="ij"), axis=-1, name='stack')  # WxHx2

        return  tf.expand_dims(tf.transpose(base_grid,[1,0,2]), axis=0)

    def spatial_expec(self,input,normalized_coordinates):

        batch_size, channels, height, width = input.shape

        # Create coordinates grid.
        grid = self.create_mesh(height, width, normalized_coordinates)
        #[1,H,W,2]
        # grid = grid.to(input.dtype)

        pos_x = tf.convert_to_tensor(tf.reshape(grid[..., 0],-1))
        pos_y = tf.convert_to_tensor(tf.reshape(grid[..., 1],-1))
        # pos_x: torch.Tensor = grid[..., 0].reshape(-1)
        # pos_y: torch.Tensor = grid[..., 1].reshape(-1)
        
        input_flat =  tf.reshape(input,(batch_size, channels, -1))#input.view(batch_size, channels, -1)

        # Compute the expectation of the coordinates.
        expected_y = tf.math.reduce_sum(tf.cast(pos_y,tf.double) * tf.cast(input_flat,tf.double), -1, keepdims=True)
        expected_x = tf.math.reduce_sum(tf.cast(pos_x,tf.double) * tf.cast(input_flat,tf.double), -1, keepdims=True)

        output = tf.concat([expected_x, expected_y], -1)

       # BxNx2
        return  tf.reshape(output,(batch_size, channels, 2))#output.view(batch_size, channels, 2)  
