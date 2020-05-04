import datetime
import wfdb
import pywt
import numpy as np
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(300, 1)),
    # 第一个卷积层, 4 个 21x1 卷积核
    tf.keras.layers.Conv1D(filters=4, kernel_size=21, strides=1, padding='SAME', activation='relu'),
    # 第一个池化层, 最大池化,4 个 3x1 卷积核, 步长为 2
    tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
    # 第二个卷积层, 16 个 23x1 卷积核
    tf.keras.layers.Conv1D(filters=16, kernel_size=23, strides=1, padding='SAME', activation='relu'),
    # 第二个池化层, 最大池化,4 个 3x1 卷积核, 步长为 2
    tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
    # 第三个卷积层, 32 个 25x1 卷积核
    tf.keras.layers.Conv1D(filters=32, kernel_size=25, strides=1, padding='SAME', activation='relu'),
    # 第三个池化层, 平均池化,4 个 3x1 卷积核, 步长为 2
    tf.keras.layers.AvgPool1D(pool_size=3, strides=2, padding='SAME'),
    # 第四个卷积层, 64 个 27x1 卷积核
    tf.keras.layers.Conv1D(filters=64, kernel_size=27, strides=1, padding='SAME', activation='relu'),
    # 打平层,方便全连接层处理
    tf.keras.layers.Flatten(),
    # 全连接层,128 个节点
    tf.keras.layers.Dense(128, activation='relu'),
    # Dropout层,dropout = 0.2
    tf.keras.layers.Dropout(rate=0.2),
    # 全连接层,5 个节点
    tf.keras.layers.Dense(5, activation='softmax')
])


# 读取心电数据和对应标签,并对数据进行小波去噪
def getDataSet(number, X_data, Y_data):
    ecgClassSet = ['N', 'A', 'V', 'L', 'R']

    # 读取心电数据记录
    print("正在读取 " + number + " 号心电数据...")
    record = wfdb.rdrecord('ecg_data/' + number, channel_names=['MLII'])
    ecg = record.p_signal
    data = []
    for i in range(len(ecg) - 1):
        Y = float(ecg[i])
        data.append(Y)

    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')

    # 获取心电数据记录中R波的位置和对应的标签
    annotation = wfdb.rdann('../ecg_data/' + number, 'atr')
    Rlocation = annotation.sample
    Rclass = annotation.symbol

    # 去掉前后的不稳定数据
    start = 10
    end = 5
    i = start
    j = len(annotation.symbol) - end

    # 因为只选择NAVLR五种心电类型,所以要选出该条记录中所需要的那些带有特定标签的数据,舍弃其余标签的点
    # X_data在R波前后截取长度为300的数据点
    # Y_data将NAVLR按顺序转换为12345
    while i < j:
        try:
            lable = ecgClassSet.index(Rclass[i])
            x_train = rdata[Rlocation[i] - 99:Rlocation[i] + 201]
            X_data.append(x_train)
            Y_data.append(lable)
            i += 1
        except ValueError:
            i += 1
    return


def loadData():
    numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                 '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                 '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                 '231', '232', '233', '234']
    dataSet = []
    lableSet = []
    for n in numberSet:
        getDataSet(n, dataSet, lableSet)

    # 转numpy数组,打乱顺序
    dataSet = np.array(dataSet).reshape(-1, 300)
    lableSet = np.array(lableSet).reshape(-1, 1)
    train_ds = np.hstack((dataSet, lableSet))
    np.random.shuffle(train_ds)

    # 返回数据集集和标签集
    X = train_ds[:, :300].reshape(-1, 300, 1)
    Y = train_ds[:, 300]
    return X, Y


def main():
    X, Y = loadData()
    # 定义日志目录,必须是启动web应用时指定目录的子目录,建议使用日期时间作为子目录名
    log_dir = "D:/python/mit-bih_ecg_prediction/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # 定义TensorBoard对象
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X, Y, epochs=50,
              batch_size=128,
              validation_split=0.3,
              callbacks=[tensorboard_callback])


if __name__ == '__main__':
    main()
