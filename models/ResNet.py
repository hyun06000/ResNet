import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Add
import datetime


class ResNet(tf.keras.Model):

    def __init__(self):
        super(ResNet, self).__init__()
        
        # =====================================================
        # =                  ----------------------------------
        # = H-PARAM & CONFIG ==================================
        # =                  ----------------------------------
        # =====================================================
        
        NUM_OF_CLASS        = 10
        USE_BIAS            = True
        EPSILON             = 1e-3
        
        KERNEL_INITIALIZER  = 'he_uniform'
        KERNEL_REGULARIZER  = tf.keras.regularizers.l2(0.0001)
        BIAS_REGULARIZER    = tf.keras.regularizers.l2(0.0001)
        BETA_REGULARIZER    = tf.keras.regularizers.l2(0.0)
        GAMMA_REGULARIZER   = tf.keras.regularizers.l2(0.0)
        
        # =====================================================
        # =        --------------------------------------------
        # = CONV_1 ============================================
        # =        --------------------------------------------
        # =====================================================
        self.conv_1_1 = tf.keras.layers.Conv2D(
            filters            = 16,
            kernel_size        = (3, 3),
            strides            = (1, 1),
            padding            = 'same',
            activation         = None,
            use_bias           = USE_BIAS,
            kernel_initializer = KERNEL_INITIALIZER,
            kernel_regularizer = KERNEL_REGULARIZER,
            bias_regularizer   = BIAS_REGULARIZER,
        )
        
        self.bn_1_1 = tf.keras.layers.BatchNormalization(
            epsilon           = EPSILON,
            beta_regularizer  = BETA_REGULARIZER,
            gamma_regularizer = GAMMA_REGULARIZER, 
        )
        
        self.relu_1_1 = tf.keras.layers.ReLU()
        
        # =====================================================
        # =        --------------------------------------------
        # = CONV_2 ============================================
        # =        --------------------------------------------
        # =====================================================
        self.conv_2_1, self.bn_2_1 = [], []
        self.conv_2_2, self.bn_2_2 = [], []
        self.relu_2_1, self.relu_2_2 =[], []
        self.shortcut_connect_2 = []
        for _ in range(3):
            self.conv_2_1.append(
                tf.keras.layers.Conv2D(
                    filters            = 16,
                    kernel_size        = (3, 3),
                    strides            = (1, 1),
                    padding            = 'same',
                    activation         = None,
                    use_bias           = USE_BIAS,
                    kernel_initializer = KERNEL_INITIALIZER,
                    kernel_regularizer = KERNEL_REGULARIZER,
                    bias_regularizer   = BIAS_REGULARIZER,
                )
            )
            
            self.bn_2_1.append(
                tf.keras.layers.BatchNormalization(
                    epsilon=EPSILON,
                    beta_regularizer=BETA_REGULARIZER,
                    gamma_regularizer=GAMMA_REGULARIZER, 
                )
            )
            self.relu_2_1.append(
                tf.keras.layers.ReLU()
            )
            
            
            self.conv_2_2.append(
                tf.keras.layers.Conv2D(
                    filters            = 16,
                    kernel_size        = (3, 3),
                    strides            = (1, 1),
                    padding            = 'same',
                    activation         = None,
                    use_bias           = USE_BIAS,
                    kernel_initializer = KERNEL_INITIALIZER,
                    kernel_regularizer = KERNEL_REGULARIZER,
                    bias_regularizer   = BIAS_REGULARIZER,
                )
            )
            
            self.bn_2_2.append(
                tf.keras.layers.BatchNormalization(
                    epsilon=EPSILON,
                    beta_regularizer=BETA_REGULARIZER,
                    gamma_regularizer=GAMMA_REGULARIZER, 
                )
            )
        
            self.relu_2_2.append(
                tf.keras.layers.ReLU()
            )
            
            self.shortcut_connect_2.append(
                tf.keras.layers.Add()
            )
            
        # =====================================================
        # =        --------------------------------------------
        # = CONV_3 ============================================
        # =        --------------------------------------------
        # =====================================================
        self.conv_3_shortcut = tf.keras.layers.Conv2D(
            filters            = 32,
            kernel_size        = (1, 1),
            strides            = (2, 2),
            padding            = 'same',
            activation         = None,
            use_bias           = USE_BIAS,
            kernel_initializer = KERNEL_INITIALIZER,
            kernel_regularizer = KERNEL_REGULARIZER,
            bias_regularizer   = BIAS_REGULARIZER,
        )
        self.relu_3_shortcut = tf.keras.layers.ReLU()
        
        self.conv_3_1, self.bn_3_1 = [], []
        self.conv_3_2, self.bn_3_2 = [], []
        self.relu_3_1, self.relu_3_2 =[], []
        self.shortcut_connect_3 = []
        for i in range(3):
            self.conv_3_1.append(
                tf.keras.layers.Conv2D(
                    filters            = 32,
                    kernel_size        = (3, 3),
                    strides            = (2, 2) if not i else (1, 1),
                    padding            = 'same',
                    activation         = None,
                    use_bias           = USE_BIAS,
                    kernel_initializer = KERNEL_INITIALIZER,
                    kernel_regularizer = KERNEL_REGULARIZER,
                    bias_regularizer   = BIAS_REGULARIZER,
                )
            )
            self.relu_3_1.append(
                tf.keras.layers.ReLU()
            )
            self.bn_3_1.append(
                tf.keras.layers.BatchNormalization(
                    epsilon=EPSILON,
                    beta_regularizer=BETA_REGULARIZER,
                    gamma_regularizer=GAMMA_REGULARIZER, 
                )
            )
            
            
            self.conv_3_2.append(
                tf.keras.layers.Conv2D(
                    filters            = 32,
                    kernel_size        = (3, 3),
                    strides            = (1, 1),
                    padding            = 'same',
                    activation         = None,
                    use_bias           = USE_BIAS,
                    kernel_initializer = KERNEL_INITIALIZER,
                    kernel_regularizer = KERNEL_REGULARIZER,
                    bias_regularizer   = BIAS_REGULARIZER,
                )
            )
            self.bn_3_2.append(
                tf.keras.layers.BatchNormalization(
                    epsilon=EPSILON,
                    beta_regularizer=BETA_REGULARIZER,
                    gamma_regularizer=GAMMA_REGULARIZER, 
                )
            )
            self.relu_3_2.append(
                tf.keras.layers.ReLU()
            )
            
            self.shortcut_connect_3.append(
                tf.keras.layers.Add()
            )
            
        # =====================================================
        # =        --------------------------------------------
        # = CONV_4 ============================================
        # =        --------------------------------------------
        # =====================================================
        self.conv_4_shortcut = tf.keras.layers.Conv2D(
            filters            = 64,
            kernel_size        = (1, 1),
            strides            = (2, 2),
            padding            = 'same',
            activation         = None,
            use_bias           = USE_BIAS,
            kernel_initializer = KERNEL_INITIALIZER,
            kernel_regularizer = KERNEL_REGULARIZER,
            bias_regularizer   = BIAS_REGULARIZER,
        )
        self.relu_4_shortcut = tf.keras.layers.ReLU()
        
        
        self.conv_4_1, self.bn_4_1 = [], []
        self.conv_4_2, self.bn_4_2 = [], []
        self.relu_4_1, self.relu_4_2 =[], []
        self.shortcut_connect_4 = []
        for i in range(3):
            self.conv_4_1.append(
                tf.keras.layers.Conv2D(
                    filters            = 64,
                    kernel_size        = (3, 3),
                    strides            = (2, 2) if not i else (1, 1),
                    padding            = 'same',
                    activation         = None,
                    use_bias           = USE_BIAS,
                    kernel_initializer = KERNEL_INITIALIZER,
                    kernel_regularizer = KERNEL_REGULARIZER,
                    bias_regularizer   = BIAS_REGULARIZER,
                )
            )
            self.bn_4_1.append(
                tf.keras.layers.BatchNormalization(
                    epsilon=EPSILON,
                    beta_regularizer=BETA_REGULARIZER,
                    gamma_regularizer=GAMMA_REGULARIZER, 
                )
            )
            self.relu_4_1.append(
                tf.keras.layers.ReLU()
            )
            
            
            self.conv_4_2.append(
                tf.keras.layers.Conv2D(
                    filters            = 64,
                    kernel_size        = (3, 3),
                    strides            = (1, 1),
                    padding            = 'same',
                    activation         = None,
                    use_bias           = USE_BIAS,
                    kernel_initializer = KERNEL_INITIALIZER,
                    kernel_regularizer = KERNEL_REGULARIZER,
                    bias_regularizer   = BIAS_REGULARIZER,
                )
            )
            self.bn_4_2.append(
                tf.keras.layers.BatchNormalization(
                    epsilon=EPSILON,
                    beta_regularizer=BETA_REGULARIZER,
                    gamma_regularizer=GAMMA_REGULARIZER, 
                )
            )
            self.relu_4_2.append(
                tf.keras.layers.ReLU()
            )
            
            self.shortcut_connect_4.append(
                tf.keras.layers.Add()
            )
            
        # =====================================================
        # =       ---------------------------------------------
        # = DENSE =============================================
        # =       ---------------------------------------------
        # =====================================================
        
        self.avg_pooling = GlobalAveragePooling2D()
        self.dense = Dense(
            units              = NUM_OF_CLASS,
            activation         = 'softmax',
            use_bias           = USE_BIAS,
            kernel_initializer = KERNEL_INITIALIZER,
            kernel_regularizer = KERNEL_REGULARIZER,
            bias_regularizer   = BIAS_REGULARIZER
        )
        
        
        
        #self.flatten = tf.keras.layers.Flatten()
        
        
        # =====================================================
        # =                ------------------------------------
        # = LAYER SET DONE ====================================
        # =                ------------------------------------
        # =====================================================
        
    
    @tf.function
    def call(self, inputs, training=False):
        
        x = inputs
        
        
        # =====================================================
        # =        --------------------------------------------
        # = CONV_1 ============================================
        # =        --------------------------------------------
        # =====================================================
        
        x = self.conv_1_1(inputs)
        x = self.bn_1_1(x, training)
        x = self.relu_1_1(x)
        
        
        
        # =====================================================
        # =        --------------------------------------------
        # = CONV_2 ============================================
        # =        --------------------------------------------
        # =====================================================
        
        shortcut = x
        for i in range(3):
            x = self.conv_2_1[i](x)
            x = self.bn_2_1[i](x, training)
            x = self.relu_2_1[i](x)
            
            x = self.conv_2_2[i](x)
            x = self.bn_2_2[i](x, training)
            x = self.relu_2_2[i](x)
            
            x = self.shortcut_connect_2[i]([x, shortcut])
            
            shortcut = x
        
        
        # =====================================================
        # =        --------------------------------------------
        # = CONV_3 ============================================
        # =        --------------------------------------------
        # =====================================================
                
        shortcut = self.conv_3_shortcut(shortcut)
        shortcut = self.relu_3_shortcut(shortcut)
        for i in range(3):
            x = self.conv_3_1[i](x)
            x = self.bn_3_1[i](x, training)
            x = self.relu_3_1[i](x)
            
            x = self.conv_3_2[i](x)
            x = self.bn_3_2[i](x, training)
            x = self.relu_3_2[i](x)
            
            x = self.shortcut_connect_3[i]([x, shortcut])
            
            shortcut = x
        
        
        # =====================================================
        # =        --------------------------------------------
        # = CONV_4 ============================================
        # =        --------------------------------------------
        # =====================================================
        
        shortcut = self.conv_4_shortcut(shortcut)
        shortcut = self.relu_4_shortcut(shortcut)
        for i in range(3):
            x = self.conv_4_1[i](x)
            x = self.bn_4_1[i](x, training)
            x = self.relu_4_1[i](x)
            
            x = self.conv_4_2[i](x)
            x = self.bn_4_2[i](x, training)
            x = self.relu_4_2[i](x)
            
            x = self.shortcut_connect_4[i]([x, shortcut])
            
            shortcut = x
        
        # =====================================================
        # =       ---------------------------------------------
        # = DENSE =============================================
        # =       ---------------------------------------------
        # =====================================================
        
        
        x = self.avg_pooling(x)
        x = self.dense(x)
        
        return x
    
    
    def trace_graph(self,input_shape):
        
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        graph_log_dir = 'logs/Graph/' + current_time + '/graph'
        graph_writer = tf.summary.create_file_writer(graph_log_dir)

        tf.summary.trace_on(graph=True)
        self.call(tf.zeros(input_shape))
        with graph_writer.as_default():
            tf.summary.trace_export(
                name="model_trace",
                step=0)
    