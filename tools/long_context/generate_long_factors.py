import random
import numpy as np
import matplotlib.pyplot as plt


dim = 128
extension_ratio = 128

def sigmoid(x, center=0, scale=1, slope=2, diff=0):
        x = (x-center) / slope
        y = 1.0 / (1 + np.exp(-x))
        return y * scale + 1  - diff
    
    
def get_sigmoid_factors(factor_dim, extension_ratio, SHAPE_FACTOR=2.667, CENTER_FACTOR=0.5, x_points=None):
    center=(factor_dim - 1) * CENTER_FACTOR  # 左右平移图像 [1/4, 2/4, 3/4]

    slope= SHAPE_FACTOR * (factor_dim / 32)  # 注意，dim变化，slope变化幅度只需与dim变化幅度相同，即可保持形状不变。
    scale=extension_ratio - 1                # 这里是保证dim-1时恰约为extension_ratio
    
    x_points = np.arange(0, factor_dim) if x_points is None else x_points
    
    _y_sigmoid_points = sigmoid(x_points, center=center, scale=scale, slope=slope, diff=0)
    diff = _y_sigmoid_points[0] - 1
    scale = scale**2 / (_y_sigmoid_points[-1]-diff)
    y_sigmoid_points = sigmoid(x_points, center=center, scale=scale, slope=slope, diff=diff)
    
    return y_sigmoid_points


def generate_phi3_lambda_factors(dim=64, extension_ratio=32):
    """ phi3 lambda factors implementation
    
    """
    # define search function
    interval = 1
    x_factors = [i for i in range(dim // 2)]
    lambda_factors = [
       random.uniform(
            sigmoid(x_factors[i] - 1.5 * interval, center=dim//4, scale=extension_ratio, slope=2.67),
            sigmoid(x_factors[i] + 1.5 * interval, center=dim//4, scale=extension_ratio, slope=2.67),
        )
        for i in range(dim // 2)
    ]
    # optimize m_scale
    if extension_ratio <= 2:
        # short search
        m_scale = random.uniform(1.0, 1.15)
    if extension_ratio > 2:
        # long search
        m_scale = random.uniform(1.17, 1.23)
    return lambda_factors, m_scale
# 绘制生成点
x_gen = np.linspace(0, dim//2-1, dim//2)
# y_gen = sigmoid(x_gen, center=(dim//2-1)/2.0, scale=extension_ratio, slope=(dim/64)*2.667)
# diff = y_gen[0] - 1
# y_gen = sigmoid(x_gen, center=(dim//2-1)/2.0, scale=extension_ratio+diff, slope=(dim/64)*2.667, diff=diff)
y_gen = get_sigmoid_factors(dim//2, extension_ratio)
print(y_gen.tolist())
plt.scatter(x_gen, y_gen, c='green', label='Generated Points')

# 绘制sigmoid曲线
# 生成x值
x_values = np.linspace(0, dim//2-1, 100)
# 计算Sigmoid函数值
y_values = get_sigmoid_factors(dim//2, extension_ratio, x_points=x_values)
# 绘制结果
plt.plot(x_values, y_values, label='Sigmoid Function')

# y = np.array([1.1323607100253377,1.255980027504134,1.3686379174216756,1.3649253135894333,1.6241971023995627,1.7070820400448092,2.550436793339597,2.575324998910281,3.4788180837722615,5.564674910889877,7.748782824491121,8.91541711591263,11.462845605232545,16.836994381830827,21.41959384633878,28.420270853192118,34.75446483182122,40.423385400923415,40.26026499934713,52.048701674384006,53.09250402171993,54.1765066356746,57.38095658199647,59.2616793306974,62.32286516390798,63.15149384486716,63.03587447688441,64.05333612022899,64.03016367459887,64.32333856217437,64.70809791244564,64.70784455872536])
# x = np.arange(0, len(y))

# plt.plot(x, y, 'ro', label='search points')

plt.xlabel('x')
plt.ylabel('Sigmoid(x)')
plt.title('Sigmoid Function')
plt.legend()
plt.grid()
# plt.savefig('sigmoid.png')

y_gen_str_list = [str(y) for y in y_gen.tolist()]
y_gen_str_list = ' '.join(y_gen_str_list)
print(y_gen_str_list)