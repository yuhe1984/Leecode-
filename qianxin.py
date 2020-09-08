# n = int(input())
# m = int(input())
# res = []
# for i in range(m):
#     num = list(map(int, input().split(' ')))
#     res.append(num)
n = 100
m = 5
res = [[22, 22], [77, 92], [29, 36], [50, 46], [99, 90]]
maxx = 0
for i in range(m):
    total = res[i][1]
    money = res[i][0]
    
    for j in range(i+1, m):
        if money + res[j][0] <= n:
            total += res[j][1]
            money += res[j][0]
        maxx = max(maxx, total)
print(maxx)