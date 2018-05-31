import numpy as np

class GRU_Helper():

    def __init__(self, **params):
        # Match the Lotus/CNTK behavior
        # If False use the python from the ONNX spec
        self.match_lotus = True

        required_inputs = ['X', 'W', 'R']
        for i in required_inputs:
            assert i in params, "Missing Required Input: {0}".format(i)

        num_directions = params['W'].shape[0]
        sequence_length = params['X'].shape[0]

        hidden_size = params['R'].shape[-1]
        batch_size = params['X'].shape[1]

        X = params['X']
        W = params['W']
        R = params['R']
        B = params['B'] if 'B' in params else np.zeros(num_directions * 6 * hidden_size).reshape(num_directions, 6 * hidden_size)
        H_0 = params['initial_h'] if 'initial_h' in params else np.zeros((num_directions, batch_size, hidden_size)).reshape(num_directions, batch_size, hidden_size)
        LBR = params['linear_before_reset'] if 'linear_before_reset' in params else 0
        self.direction = params['direction'] if 'direction' in params else 'forward'

        if (num_directions == 1):
            if (self.direction == 'forward'):
                self.one = OneDirectionGRU(X, W, R, B, H_0, LBR)
            else:
                # flip input so we process in reverse
                self.one = OneDirectionGRU(np.flip(X,0), W, R, B, H_0, LBR)

            self.two = None

        else:
            # split the inputs which have per direction rows
            Wfw, Wbw = np.vsplit(W, 2)
            Rfw, Rbw = np.vsplit(R, 2)
            Bfw, Bbw = np.vsplit(B, 2)
            H_0fw, H_0bw = np.vsplit(H_0, 2)

            self.one = OneDirectionGRU(X, Wfw, Rfw, Bfw, H_0fw, LBR)
            self.two = OneDirectionGRU(np.flip(X, 0), Wbw, Rbw, Bbw, H_0bw, LBR)

    def run(self):

        if (self.direction == 'bidirectional'):
            f_output = self.one.execute()
            r_output = self.two.execute()

            # flip reverse output it matches the original input order
            r_output_orig_input_order = np.flip(r_output, 0)

            # create merged output by merging the forward and reverse rows for seq_length
            # 0 rows, 2 directions, batch size, hidden_size
            seq_length = f_output.shape[0]
            batch_size = f_output.shape[2]
            hidden_size = f_output.shape[3]

            output = np.empty((0, 2, batch_size, hidden_size), np.float32)
            for x in range(0, seq_length):
                output = np.append(output, f_output[x])
                output = np.append(output, r_output_orig_input_order[x])

            output = output.reshape(seq_length, 2, batch_size, hidden_size)
        else:
            output = self.one.execute()
            if (self.direction == 'reverse'):
                # flip so it's back in the original order of the inputs
                output = np.flip(output, 0)

        return output

class OneDirectionGRU():

    def __init__(self, X, W, R, B, initial_h, LBR):

        # set debug to True to dump most of the tensors from the various calculations
        self.debug = False
        self.match_lotus = True
        self.X = X
        # remove num_directions axis for W, R, B, H_0
        self.W = np.squeeze(W, axis=0)
        self.R = np.squeeze(R, axis=0)
        self.B = np.squeeze(B, axis=0)
        self.H_0 = np.squeeze(initial_h, axis=0)
        self.LBR = LBR

    def f(self, x):
        return 1 / (1 + np.exp(-x))

    def g(self, x):
        return np.tanh(x)

    def print_with_shape(self, name, a):
        if (self.debug):
            print(name + " [shape: ", a.shape, "]\n", a)

    def execute(self):
        self.print_with_shape("X", self.X)

        [w_z, w_r, w_h] = np.split(self.W, 3)
        [r_z, r_r, r_h] = np.split(self.R, 3)
        [w_bz, w_br, w_bh, r_bz, r_br, r_bh] = np.split(self.B, 6)

        #self.print_with_shape("w_z", w_z)
        #self.print_with_shape("w_r", w_r)
        #self.print_with_shape("w_h", w_h)

        #self.print_with_shape("r_z", r_z)
        #self.print_with_shape("r_r", r_r)
        #self.print_with_shape("r_h", r_h)

        #self.print_with_shape("w_bz", w_bz)
        #self.print_with_shape("w_br", w_br)
        #self.print_with_shape("w_bh", w_bh)
        #self.print_with_shape("r_bz", r_bz)
        #self.print_with_shape("r_br", r_br)
        #self.print_with_shape("r_bh", r_bh)

        seq_len = self.X.shape[0]
        num_directions = 1
        hidden_size = self.R.shape[-1]
        batch_size = self.X.shape[1]

        output = np.empty((0, num_directions, batch_size, hidden_size), np.float32)

        for row in self.X:
            self.print_with_shape ("row", row)
            wz_t = np.transpose(w_z)
            #self.print_with_shape ("wz_t", wz_t)
            r_wzt = (np.dot(row, wz_t))
            self.print_with_shape ("r_wzt", r_wzt)

            wr_t = np.transpose(w_r)
            #self.print_with_shape ("wr_t", wr_t)
            r_wrt = (np.dot(row, wr_t))
            self.print_with_shape ("r_wrt", r_wrt)

            if self.match_lotus:
                h0_rz = np.dot(self.H_0, np.transpose(r_z))
                h0_rr = np.dot(self.H_0, np.transpose(r_r))
                wz_h0_rz = np.dot(row, np.transpose(w_z)) + h0_rz
                wr_h0_rr = np.dot(row, np.transpose(w_r)) + h0_rr
                z = self.f(wz_h0_rz + w_bz + r_bz)
                r = self.f(wr_h0_rr + w_br + r_br)
                h_default = self.g(np.dot(row, np.transpose(w_h)) + np.dot(r * self.H_0, np.transpose(r_h)) + w_bh + r_bh)
                h_linear = self.g(np.dot(row, np.transpose(w_h)) + r * (np.dot(self.H_0, np.transpose(r_h)) + r_bh) + w_bh)

            else:
                # this is what the ONNX example has. there is no transpose of R[zrh]
                h0_rz = np.dot(self.H_0, r_z)
                h0_rr = np.dot(self.H_0, r_r)
                wz_h0_rz = np.dot(row, np.transpose(w_z)) + h0_rz
                wr_h0_rr = np.dot(row, np.transpose(w_r)) + h0_rr
                z = self.f(wz_h0_rz + w_bz + r_bz)
                r = self.f(wr_h0_rr + w_br + r_br)
                h_default = self.g(np.dot(row, np.transpose(w_h)) + np.dot(r * self.H_0, r_h) + w_bh + r_bh)
                h_linear = self.g(np.dot(row, np.transpose(w_h)) + r * (np.dot(self.H_0, r_h) + r_bh) + w_bh)

            h = h_linear if self.LBR else h_default

            self.print_with_shape("h0_rz", h0_rz)
            self.print_with_shape("h0_rr", h0_rr)
            self.print_with_shape("wz_h0_rz", wz_h0_rz)
            self.print_with_shape("wr_h0_rr", wr_h0_rr)
            self.print_with_shape("z", z)
            self.print_with_shape("r", r)
            self.print_with_shape("h", h)

            H = (1 - z) * h + z * self.H_0

            self.print_with_shape("H", H)
            output = np.append(output, H.reshape(1, 1, batch_size, hidden_size), axis=0)

            self.H_0 = H

        return output

class LotusRTTestContext():

    @staticmethod
    def OneDirectionWeights():

        hidden_size = 2

        W = np.array([[[-0.494659, 0.0453352],  # Wz
                       [-0.487793, 0.417264],
                       [-0.0091708, -0.255364],  # Wr
                       [-0.106952, -0.266717],
                       [-0.0888852, -0.428709],  # Wh
                       [-0.283349, 0.208792]]]).astype(np.float32)

        R = np.array([[[0.146626, -0.0620289],  # Rz
                       [-0.0815302, 0.100482],
                       [-0.228172, 0.405972],  # Rr
                       [0.31576, 0.281487],
                       [-0.394864, 0.42111],  # Rh
                       [-0.386624, -0.390225]]]).astype(np.float32)

        W_B = np.array([[0.381619, 0.0323954,  # Wbz
                         -0.258721, 0.45056,  # Wbr
                         -0.250755, 0.0967895]]).astype(np.float32)  # Wbh
        R_B = np.zeros((1, 3 * hidden_size)).astype(np.float32)
        B = np.concatenate((W_B, R_B), axis=1)

        return W, R, B

    @staticmethod
    def BidirectionalWeights():

        W1, R1, B1 = LotusRTTestContext.OneDirectionWeights()

        hidden_size = R1.shape[-1]
        input_size = W1.shape[-1]

        W = np.tile(W1, (2, 1)).reshape(2, 3 * hidden_size, input_size)
        R = np.tile(R1, (2, 1)).reshape(2, 3 * hidden_size, hidden_size)
        B = np.tile(B1, (2,1))

        return W, R, B

# replicate Lotus unit tests inputs to validate output
class GRU_LotusUnitTests():

    @staticmethod
    def ForwardDefaultActivationsSimpleWeightsNoBiasTwoRows():

        print(GRU_LotusUnitTests.ForwardDefaultActivationsSimpleWeightsNoBiasTwoRows.__name__)

        seq_length = 2
        batch_size = 2
        input_size = 1
        hidden_size = 3
        input = np.array([[[1.], [2.]],
                          [[10.], [11.]]]).astype(np.float32)


        W = np.array([0.1, 0.2, 0.3, 1, 2, 3, 10, 11, 12]).astype(np.float32).reshape(1, 3 * hidden_size, input_size)

        weight_scale = 0.1
        R = weight_scale * np.ones((1, 3 * hidden_size, hidden_size)).astype(np.float32)

        gru = GRU_Helper(X=input,W=W,R=R,direction='forward')
        fw_output = gru.run()
        print (fw_output)

    @staticmethod
    def ReverseDefaultActivationsSimpleWeightsNoBiasTwoRows():

        print(GRU_LotusUnitTests.ReverseDefaultActivationsSimpleWeightsNoBiasTwoRows.__name__)

        seq_length = 2
        batch_size = 2
        input_size = 1
        hidden_size = 3
        input = np.array([[[1.], [2.]],
                          [[10.], [11.]]]).astype(np.float32)


        W = np.array([0.1, 0.2, 0.3, 1, 2, 3, 10, 11, 12]).astype(np.float32).reshape(1, 3 * hidden_size, input_size)

        weight_scale = 0.1
        R = weight_scale * np.ones((1, 3 * hidden_size, hidden_size)).astype(np.float32)

        gru = GRU_Helper(X=input,W=W,R=R,direction='reverse')
        fw_output = gru.run()
        print(fw_output)

    @staticmethod
    def BidirectionalDefaultActivationsSimpleWeightsNoBiasTwoRows():

        print(GRU_LotusUnitTests.BidirectionalDefaultActivationsSimpleWeightsNoBiasTwoRows.__name__)

        seq_length = 2
        batch_size = 2
        input_size = 1
        hidden_size = 3

        input = np.array([[[1.], [2.]],
                          [[10.], [11.]]]).astype(np.float32)

        W = np.array([0.1, 0.2, 0.3, 1, 2, 3, 10, 11, 12]).astype(np.float32).reshape(1, 3 * hidden_size, input_size)

        weight_scale = 0.1
        R = weight_scale * np.ones((1, 3 * hidden_size, hidden_size)).astype(np.float32)

        # duplicate the W and R inputs so we use the same values for both forward and reverse
        gru = GRU_Helper(X=input,
                         W=np.tile(W,(2,1)).reshape(2, 3 * hidden_size, input_size),
                         R=np.tile(R,(2,1)).reshape(2, 3 * hidden_size, hidden_size),
                         direction='bidirectional')

        fw_output = gru.run()
        print(fw_output)

    @staticmethod
    def Legacy_TestGRUOpForwardBasic():

        print(GRU_LotusUnitTests.Legacy_TestGRUOpForwardBasic.__name__)

        input = np.array([[[-0.455351, -0.276391]],
                          [[-0.185934, -0.269585]]]).astype(np.float32)

        W, R, B = LotusRTTestContext.OneDirectionWeights()
        gru = GRU_Helper(X=input, W=W, R=R, B=B)
        output = gru.run()
        print(output)

    @staticmethod
    def Legacy_TestGRUOpBackwardBasic():
        print(GRU_LotusUnitTests.Legacy_TestGRUOpBackwardBasic.__name__)

        input = np.array([[[-0.185934, -0.269585]],
                          [[-0.455351, -0.276391]]]).astype(np.float32)

        W, R, B = LotusRTTestContext.OneDirectionWeights()
        gru = GRU_Helper(X=input, W=W, R=R, B=B, direction='reverse')
        output = gru.run()
        print(output)

    @staticmethod
    def Legacy_TestGRUOpBidirectionalBasic():

        print(GRU_LotusUnitTests.Legacy_TestGRUOpBidirectionalBasic.__name__)

        input = np.array([[[-0.455351, -0.276391]],
                          [[-0.185934, -0.269585]]]).astype(np.float32)

        W, R, B = LotusRTTestContext.BidirectionalWeights()
        gru = GRU_Helper(X=input, W=W, R=R, B=B, direction='bidirectional')
        output = gru.run()
        print(output)


GRU_LotusUnitTests.ForwardDefaultActivationsSimpleWeightsNoBiasTwoRows()
GRU_LotusUnitTests.ReverseDefaultActivationsSimpleWeightsNoBiasTwoRows()
GRU_LotusUnitTests.BidirectionalDefaultActivationsSimpleWeightsNoBiasTwoRows()
GRU_LotusUnitTests.Legacy_TestGRUOpForwardBasic()
GRU_LotusUnitTests.Legacy_TestGRUOpBackwardBasic()
GRU_LotusUnitTests.Legacy_TestGRUOpBidirectionalBasic()