import xlrd

# mi ni 连续增长/下降的次数
m2 = 0
m3 = 0
m4 = 0
m5 = 0
n2 = 0
n3 = 0
n4 = 0
n5 = 0

def constant_droporincrease_rate(rows_rate,lst_m,lst_n,num):
    # 计算增长/下降 0.9 0.1的比例
    de_val_four = []
    in_val_four = []
    for item in lst_m:
        temp = 0
        for i in range(num):
            temp += rows_rate[item-i]
        de_val_four.append(temp)
    for item in lst_n:
        temp = 0
        for i in range(num):
            temp += rows_rate[item-i]
        in_val_four.append(temp)
    de_val_four.sort()
    in_val_four.sort()
    de_m_four_half = de_val_four[int(len(de_val_four)*0.1)]
    in_m_four_half = in_val_four[int(len(de_val_four)*0.9)]
    print("第",num,"次",sum(de_val_four))
    print("第",num,"次",sum(in_val_four))
    return de_m_four_half, in_m_four_half

def func(lst,num,mode):
    lst2 = []
    ans = 0
    length = len(lst)
    # 单调递减
    if mode == True:
        temp = 0
        for i in range(length):
            if lst[i] == 0:
                temp += 1
                if (temp == num and i == length-1) or (temp == num and i!= length-1 and lst[i+1] == 1):
                    lst2.append(i)
                    ans += 1
                    temp = 0
            else:
                temp = 0
    # 单调递增
    else:
        temp = 0
        for i in range(length):
            if lst[i] == 1:
                temp += 1
                if (temp == num and i == length-1) or (temp == num and i!= length-1 and lst[i+1] == 0):
                    lst2.append(i)
                    ans += 1
                    temp = 0
            else:
                temp = 0
    return ans, lst2


def read_file(path):
    sheets = xlrd.open_workbook(path)
    sheet1 = sheets.sheet_by_index(0)
    cols = sheet1.col_values(3)
    cols2 = sheet1.col_values(2)
    return cols[2:], cols2[2:]

def calculate_m(path):
    sheets = xlrd.open_workbook(path)
    sheet1 = sheets.sheet_by_index(0)
    cols = sheet1.col_values(2)
    cols = cols[2:]
    de_cols = []
    in_cols = []
    for item in cols:
        # 下降
        if item < 0:
            de_cols.append(item)
        else:
            in_cols.append(item)
    de_cols.sort()
    in_cols.sort()
    de_len = len(de_cols)
    in_len = len(in_cols)
    de_index = int(de_len * 0.1)
    de_index_half = int(de_len * 0.5)
    in_index = int(in_len * 0.9)
    in_index_half = int(in_len * 0.5)
    de_m = de_cols[de_index]
    de_m_half = de_cols[de_index_half]
    in_m = in_cols[in_index]
    in_m_half = in_cols[in_index_half]
    return de_m, de_m_half, in_m, in_m_half




def main():
    content_path = "gold_rate.xls"
    content_path2 = "bitcoin_rate.xls"
    rows,rows_rate = read_file(content_path)
    print("gold")
    m2,lst_m2 = func(rows,2,True)
    m3,lst_m3 = func(rows,3,True)
    m4,lst_m4 = func(rows,4,True)
    m5,lst_m5 = func(rows,5,True)
    m6,_ = func(rows,6,True)
    n2,lst_n2 = func(rows,2,False)
    n3,lst_n3 = func(rows,3,False)
    n4,lst_n4 = func(rows,4,False)
    n5,lst_n5 = func(rows,5,False)
    n6,_ = func(rows,6,False)
    # print(func([1,1,1,1,0,1,1,1],2,True))
    print("单调递减2-6次",m2,m3,m4,m5,m6)
    print("单调递增2-6次",n2,n3,n4,n5,n6)
    # print("lst_m4",lst_m4)
    # print("lst_n4",lst_n4)
    print("bitcoin")
    rows,_ = read_file(content_path2)
    m2,_ = func(rows,2,True)
    m3,_ = func(rows,3,True)
    m4,lst_m4_bit = func(rows,4,True)
    m5,_ = func(rows,5,True)
    m6,_ = func(rows,6,True)
    n2,_ = func(rows,2,False)
    n3,_ = func(rows,3,False)
    n4,lst_n4_bit = func(rows,4,False)
    n5,_ = func(rows,5,False)
    n6,_ = func(rows,6,False)
    # print(func([1,1,1,1,0,1,1,1],2,True))
    print("单调递减2-6次",m2,m3,m4,m5,m6)
    print("单调递增2-6次",n2,n3,n4,n5,n6)
    print("*"*20)
    print("M值计算")
    print("gold")
    # 单次平均计算
    de_m, de_m_half, in_m, in_m_half = calculate_m(content_path)
    # 连续n次平均计算
    # print("rows_rate",rows_rate)
    # print("lst_m2",lst_m2)
    # print("lst_m3",lst_m3)
    # print("lst_m4",lst_m4)
    # print("lst_n2",lst_n2)
    # print("lst_n3",lst_n3)
    # print("lst_n4",lst_n4)
    de_m_two_half, de_n_two_half = constant_droporincrease_rate(rows_rate,lst_m2,lst_n2,2)
    de_m_three_half, de_n_three_half = constant_droporincrease_rate(rows_rate,lst_m3,lst_n3,3)
    de_m_four_half, de_n_four_half = constant_droporincrease_rate(rows_rate,lst_m4,lst_n4,4)
    de_m_five_half, de_n_five_half = constant_droporincrease_rate(rows_rate,lst_m5,lst_n5,5)
    print("跌幅")
    print("M0.1_总", de_m)
    print("M0.1_连续二次", de_m_two_half)
    print("M0.1_连续三次", de_m_three_half)
    print("M0.1_连续四次", de_m_four_half)
    print("M0.1_连续五次", de_m_five_half)
    print("M0.5_总", de_m_half)
    print("涨幅")
    print("M0.9_总", in_m)
    print("M0.9_连续二次", de_n_two_half)
    print("M0.9_连续三次", de_n_three_half)
    print("M0.9_连续四次", de_n_four_half)
    print("M0.9_连续五次", de_n_five_half)
    print("M0.5_总", in_m_half)
    print("bitcoin")
    de_m_bit, de_m_half_bit, in_m_bit, in_m_half_bit = calculate_m(content_path2)
    print("跌幅")
    print("M0.1", de_m_bit)
    print("M0.5", de_m_half_bit)
    print("涨幅")
    print("M0.9", in_m_bit)
    print("M0.5", in_m_half_bit)

if __name__ == "__main__":
    main()
# 结果
# gold
# 单调递减 71 34 15 7 10 (2,3,4,5,6次)
# 单调递增 88 36 29 8 8 (2,3,4,5,6次)

# bitcoin
# 单调递减 117 35 25 16 1 (2,3,4,5,6次)
# 单调递增 116 60 29 12 13 (2,3,4,5,6次)

# u1 = 4 (90%以上都最多连续降/升4次)

# 拟合结果
# 2次 y=2.63x
# 3次 y=2.19x z=4.82x
# 4次 y=3.12x z=9.76x t=30.54x
# 确定p0 500美金本金 20美金投入 平均