import xlrd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 常量
bitcoin = 0
cash = 500
gold = 0
preperty = 500
days = []
preperties = [500,500,500,500,500]
rate3 = 0.01
rate2 = 1 - rate3

def read_file(path):
    sheets = xlrd.open_workbook(path)
    sheet1 = sheets.sheet_by_index(0)
    cols = sheet1.col_values(1)[1:] # 真实值
    cols2 = sheet1.col_values(3)[1:] # 预测值
    cols2_half = sheet1.col_values(4)[1:] # 预测值
    cols3 = sheet1.col_values(5)[1:] # change rate
    cols4 = sheet1.col_values(6)[1:] # increase/de 1/0
    cols44 = sheet1.col_values(8)[6:-1] # increase/de 1/0
    cols5 = [0] # 连续跌n次 -n
    cols6 = [0] # 幅度
    length = len(cols[1:])
    for i in range(1,length):
        temp = int(cols4[i])
        state = 1 if temp == 1 else -1
        j = i
        while j >= 0:
            j = j - 1
            if (float(cols4[j]) * state > 0) or (state < 0 and cols4[j] == 0):
                if state > 0:
                    state =  state + 1
                else:
                    state =  state - 1
            else:
                break
        cols5.append(state)
        val = 0
        for q in range(abs(state)):
            val = val + cols3[i-q]
        cols6.append(val)

    return cols, cols2, cols3, cols4, cols5, cols6, cols2_half, cols44 # 前5个预测为0

def func(path):
    global cash
    global gold
    global days
    global rate2
    global rate3
    # col5 -> 连续跌n次 -n col6 累计变化率
    money = 0
    day = 0
    col1, col2, col3, col4, col5, col6, col_predict, col_predict_rate = read_file(path)
    col3.sort()
    length = len(col1)
    flag = 0
    continue_list = []
    for i in range(0,length-10):
        temp = cash + gold * col1[i+5]
        preperties.append(temp)
        q = i + 5
        if q in continue_list:
            continue
        rate = 1
        for j in range(3):
            rate = rate * (1 + col_predict_rate[i+j])
        if col3[i+5] < 0:
            if rate > 1.01:
                flag = 1
                day = i+3
                state = col5[i+5]
                if state > 4:
                    continue
                base = 500 / (1+3+10+30)
                if state == 1:
                    change = base
                elif state == 2:
                    change = base * 3
                elif state == 3:
                    change = base * 10
                else:
                    change = base * 30
                if col3[i+5] >= 0:
                    change = change * (-1)
                if money == 500:
                    continue
                if money + change >= 0 :
                    if money + change < 500:
                        money = money + change
                    else:
                        change = 500 - money
                        money = 500

                    gold = gold + change*rate2/col1[i+5]
                    cash = cash - change
                    print("第",i+5,"天交易额",change) # 买入
                    days.append(i+5)
                continue
        if flag == 1 and i > day and money>0: # 没有卖出
            for m in range(3):
                if col3[i+m+5] < 0:
                    print("第",i+5,"天交易额",-money)
                    print("当前money",money)
                    days.append(i+5)
                    cash = cash + rate2*gold * col1[i+5]
                    gold = 0
                    money = 0
                    flag = 0

                    break
        if rate > (1+rate3*2):
            change = 0
            if money == 500:
                continue
            else:
                print("money",money)
                change = 500 - money
                print("change",change)
                money = 500
                print("第",i+5,"天交易额1",change)
                days.append(i+5)
                gold = gold + rate2*change/col1[i+5]
                cash = cash - change
                print("当前money",money)
                print("第",i+8,"天交易额2",-500)
                cash = cash + rate2 * gold * col1[i+8]
                days.append(i+8)
                gold = 0
                for n in range(i+6,i+9):
                    continue_list.append(n)
                money = 0
                i += 3
                print("当前money",money)

def theory_max(path):
    col1, col2, col3, col4, col5, col6, col_predict, col_predict_rate = read_file(path)
    ans = maxProfit(len(col1),col1)
    print(ans)

def maxProfit(k, prices):
    k = min(k, len(prices) // 2)
    buy = [-float("inf")] * (k+1)  # 买入
    sell = [0] * (k+1)  # 卖出
    for p in prices:
        for i in range(1, k+1):
            if sell[i-1]-p>buy[i] and (i ==1 or i==2):
                # print("交易1","i",i,"p",p)
                # print("***")
                # print(buy[2])
                # print(sell[2])
                pass
            buy[i] = max(buy[i], (sell[i-1] - p)*0.99)
            if buy[i]+p>sell[i] and (i ==1 or i==2):
                # print("交易2","i",i,"p",p)
                # print("***")
                # print(buy[2])
                # print(sell[2])
                pass
            sell[i] = max(sell[i], (buy[i] + p)*0.99)
    return sell[-1]


def main():
    path = "predict/gold.xls"
    func(path)
    print("cash",cash)
    print("gold",gold)
    print("days",days)
    temp = preperties[-1]
    for i in range(5):
        preperties.append(temp)
    print("properties",preperties)
    x = range(len(preperties))
    plt.plot(x,preperties)
    plt.xlabel("time/days")
    plt.ylabel("value/dollars")
    plt.title("return on gold investment")
    plt.show()
    print("*********")
    theory_max(path)
    print("资产最大值",preperties[-1])

if __name__ == "__main__":
    main()