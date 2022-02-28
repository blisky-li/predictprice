import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xlwt
import  xlrd
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']

def exponential_smoothing_1(alpha, data):
    '''
    一次指数平滑
    :param alpha:  平滑系数
    :param data:   数据序列：list
    :return:       返回一次指数平滑值：list
    '''
    s_single=[]
    s_single.append(data[0])
    for i in range(1, len(data)):
        s_single.append(alpha * data[i] + (1 - alpha) * s_single[i-1])
    return s_single

def exponential_smoothing_2(alpha, data):
    '''
    二次指数平滑
    :param alpha:  平滑系数
    :param data:   数据序列：list
    :return:       返回二次指数平滑值,参数a, b：list
    '''
    s_single = exponential_smoothing_1(alpha, data)
    s_double = exponential_smoothing_1(alpha, s_single)
    a_double = [0 for i in range(len(data))]
    b_double = [0 for i in range(len(data))]
    F_double = [0 for i in range(len(data))]
    for i in range(len(data)):
        a = 2 * s_single[i] - s_double[i]
        b = (alpha / (1 - alpha)) * (s_single[i] - s_double[i])
        F = a + b
        a_double[i] = a
        b_double[i] = b
        F_double[i] = F
    return a_double,b_double,F_double

def exponential_smoothing_3(alpha, data):
    '''
    三次指数平滑
    :param alpha:  平滑系数
    :param data:   数据序列：list
    :return:       返回二次指数平滑值，参数a, b, c，预测值Ft+1：list
    '''
    s_single = exponential_smoothing_1(alpha, data)
    s_double = exponential_smoothing_1(alpha, s_single)
    s_triple = exponential_smoothing_1(alpha, s_double)

    a_triple = [0 for i in range(len(data))]
    b_triple = [0 for i in range(len(data))]
    c_triple = [0 for i in range(len(data))]
    F_triple = [0 for i in range(len(data))]
    for i in range(len(data)):
        a = 3 * s_single[i] - 3 * s_double[i] + s_triple[i]
        b = (alpha / (2 * ((1 - alpha) ** 2))) * ((6 - 5 * alpha) * s_single[i] - 2 * ((5 - 4 * alpha) * s_double[i]) + (4 - 3 * alpha) * s_triple[i])
        c = ((alpha ** 2) / (2 * ((1 - alpha) ** 2))) * (s_single[i] - 2 * s_double[i] + s_triple[i])
        F = a + b + c
        a_triple[i] = a
        b_triple[i] = b
        c_triple[i] = c
        F_triple[i] = F
    return a_triple, b_triple, c_triple, F_triple
def model_error_analysis(F, data):
    '''
    误差分析
    :param F:     预测数列：list
    :param data:  原始序列：list
    :return:      返回各期绝对误差，相对误差：list，返回平均绝对误差和平均相对误差
    '''
    AE = [0 for i in range(len(data)-1)]
    RE = []
    AE_num = 0
    RE_num = 0
    for i in range(1,len(data)):
        _AE = abs(F[i-1] - data[i])
        _RE = _AE / data[i]
        AE_num += _AE
        RE_num += _RE
        AE[i-1] = _AE
        RE.append('{:.2f}%'.format(_RE*100))
    MAE = AE_num / (len(data)-1)
    MRE = '{:.2f}%'.format(RE_num *100 / (len(data)-1))
    return AE, MAE, RE, MRE

def alpha_analysis(data, itype=2):

    '''
    判断误差最小的平滑系数
    :param data:   原始序列：list
    :param itype:  平滑类型：1,2,3
    :return:       返回平均绝对误差最小的平滑系数和最小平均绝对误差
    '''
    alpha_all = [0.01 * i for i in range(1, 100)]  # 只需要0.1-0.9修改为alpha_triple = [0.1 * i for i in range(1,10)]
    best_alpha = 0
    min_MAE = float('Inf')  # 无穷大
    if itype == 2:
        for i in range(len(alpha_all)):
            alpha = alpha_all[i]
            a_double, b_double, F_double = exponential_smoothing_2(alpha, data)
            AE_double, MAE_double, RE_double, MRE_double = model_error_analysis(F_double, data)
            if MAE_double <= min_MAE:
                min_MAE = MAE_double
                best_alpha = alpha
            else:
                pass
    elif itype == 3:
        for i in range(len(alpha_all)):
            alpha = alpha_all[i]
            a_triple, b_triple, c_triple, F_triple = exponential_smoothing_3(alpha, data)
            AE_triple, MAE_triple, RE_triple, MRE_triple = model_error_analysis(F_triple, data)
            if MAE_triple <= min_MAE:
                min_MAE = MAE_triple
                best_alpha = alpha
            else:
                pass
    else:
        for i in range(len(alpha_all)):
            alpha = alpha_all[i]
            F_single = exponential_smoothing_1(alpha, data)
            AE_single, MAE_single, RE_single, MRE_single = model_error_analysis(F_single, data)
            if MAE_single <= min_MAE:
                min_MAE = MAE_single
                best_alpha = alpha
            else:
                pass

    return best_alpha, min_MAE

def scatter_diagram(F, data, t):
    '''
    绘制散点图
    :param F:     预测序列：list
    :param data:  原始类型：list
    :param t:     时间类型：list
    '''
    F = F[:-1:]
    data = data[1::]
    t = t[1::]
    plt.title("散点图",fontsize=20)  #图表名称
    plt.xlabel("年份", fontsize=12)  #改x坐标轴标题
    plt.ylabel("货邮吞吐量（千吨）", fontsize=12)  #改y坐标轴标题
    plt.scatter(t, data, label='实际值',s=10)
    plt.scatter(t, F, marker = 'x', label='预测值',s=10)
    plt.legend()
    plt.savefig('散点图.png', bbox_inches='tight',dpi = 300)
    plt.show()

def line_chart(F, data, t):
    '''
    绘制折现图
    :param F:     预测序列：list
    :param data:  原始类型：list
    :param t:     时间类型：list
    '''
    F = F[:-1:]
    data = data[1::]
    t = t[1::]
    plt.title("对比曲线",fontsize=20)
    plt.xlabel("年份", fontsize=12)
    plt.ylabel("货邮吞吐量（千吨）", fontsize=12)
    plt.plot(t, data, label='实际值')
    plt.plot(t, F, label='预测值')
    plt.legend()
    plt.savefig('折线图.png', bbox_inches='tight',dpi = 300)
    plt.show()


def write_xls(alpha, data, t):
    '''
    写入表格
    :param alpha:  平滑系数
    :param data:   原始类型：list
    :param t:      时间类型：list
    '''
    workbook = xlwt.Workbook()
    worksheet_1 = workbook.add_sheet('二次指数平滑')
    worksheet_2 = workbook.add_sheet('三次指数平滑')

    s_single = exponential_smoothing_1(alpha, data)
    s_double = exponential_smoothing_1(alpha, s_single)
    s_triple = exponential_smoothing_1(alpha, s_double)

    a_double, b_double, F_double = exponential_smoothing_2(alpha, data)
    AE_double, MAE_double, RE_double, MRE_double = model_error_analysis(F_double, data)
    title_1 = ['时间', 't', '实际值', '一次指数平滑值', '二次指数平滑值', 'a', 'b', 'F', '绝对误差', '相对误差']
    col = 0
    for w in title_1:
        worksheet_1.write(0, col, w)
        col += 1
    worksheet_1.write(1, 0, t[0])
    worksheet_1.write(1, 1, 1)
    worksheet_1.write(1, 2, data[0])
    worksheet_1.write(1, 3, s_single[0])
    worksheet_1.write(1, 4, s_double[0])
    worksheet_1.write(1, 5, a_double[0])
    worksheet_1.write(1, 6, b_double[0])
    row = 2
    for i in range(1, len(data)):
        worksheet_1.write(row, 0, t[i])
        worksheet_1.write(row, 1, i + 1)
        worksheet_1.write(row, 2, data[i])
        worksheet_1.write(row, 3, s_single[i])
        worksheet_1.write(row, 4, s_double[i])
        worksheet_1.write(row, 5, a_double[i])
        worksheet_1.write(row, 6, b_double[i])
        worksheet_1.write(row, 7, F_double[i - 1])
        worksheet_1.write(row, 8, AE_double[i - 1])
        worksheet_1.write(row, 9, RE_double[i - 1])
        row += 1
    worksheet_1.write_merge(row, row, 0, 8, '平均绝对误差')
    worksheet_1.write_merge(row + 1, row + 1, 0, 8, '平均相对误差')
    worksheet_1.write(row, 9, MAE_double)
    worksheet_1.write(row + 1, 9, MRE_double)

    a_triple, b_triple, c_triple, F_triple = exponential_smoothing_3(alpha, data)
    AE_triple, MAE_triple, RE_triple, MRE_triple = model_error_analysis(F_triple, data)
    title_2 = ['时间', 't', '实际值', '一次指数平滑值', '二次指数平滑值', '三次指数平滑值', 'a', 'b', 'c', 'F', '绝对误差', '相对误差']
    col = 0
    for w in title_2:
        worksheet_2.write(0, col, w)
        col += 1
    worksheet_2.write(1, 0, t[0])
    worksheet_2.write(1, 1, 1)
    worksheet_2.write(1, 2, data[0])
    worksheet_2.write(1, 3, s_single[0])
    worksheet_2.write(1, 4, s_double[0])
    worksheet_2.write(1, 5, s_triple[0])
    worksheet_2.write(1, 6, a_triple[0])
    worksheet_2.write(1, 7, b_triple[0])
    worksheet_2.write(1, 8, c_triple[0])
    row = 2
    for i in range(1, len(data)):
        worksheet_2.write(row, 0, t[i])
        worksheet_2.write(row, 1, i + 1)
        worksheet_2.write(row, 2, data[i])
        worksheet_2.write(row, 3, s_single[i])
        worksheet_2.write(row, 4, s_double[i])
        worksheet_2.write(row, 5, s_triple[i])
        worksheet_2.write(row, 6, a_triple[i])
        worksheet_2.write(row, 7, b_triple[i])
        worksheet_2.write(row, 8, c_triple[i])
        worksheet_2.write(row, 9, F_triple[i - 1])
        worksheet_2.write(row, 10, AE_triple[i - 1])
        worksheet_2.write(row, 11, RE_triple[i - 1])
        row += 1
    worksheet_2.write_merge(row, row, 0, 10, '平均绝对误差')
    worksheet_2.write_merge(row + 1, row + 1, 0, 10, '平均相对误差')
    worksheet_2.write(row, 11, MAE_triple)
    worksheet_2.write(row + 1, 11, MRE_triple)
    workbook.save('指数平滑预测.xls')


workbook = xlrd.open_workbook("gold.xls")  # 文件路径
worksheet=workbook.sheet_by_name("Sheet1")
l1 = []
l2 = []
nrows=worksheet.nrows
for i in range(nrows): #循环打印每一行
    l1.append('d'+str(int(worksheet.row_values(i)[0])))
    l2.append(float(worksheet.row_values(i)[1]))

#print(alpha_analysis(l2,itype=1))
alpha = 0.9

#print(exponential_smoothing_3(alpha, l2))
a_triple, b_triple, c_triple, F_triple = exponential_smoothing_3(alpha, l2)
a = int(len(F_triple)*0.7)
print(list(F_triple[a:]))
#line_chart(list(F_triple)[a:], l2[a:], l1[a:])
