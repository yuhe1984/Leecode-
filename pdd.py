# # # res1 = list(map(int, input().split(' ')))
# # # K, N = res1[0], res1[1]
# # # res = list(map(int, input().split(' ')))
# # # class Solution():
# # #     def sezi(self, res, K, N):
# # #         ans = 0
# # #         x = 0
# # #         for i in range(N):
# # #             if ans == K:
# # #                 return 'paradox'
# # #             elif ans < K:
# # #                 ans += res[i]
# # #                 if ans > K:
# # #                     ans = K - (ans - K)
# # #                     x += 1
# # #         return [K - ans, x]
    
    
# # # a = Solution()
# # # b = a.sezi(res, K, N)
# # # if b == 'paradox':
# # #     print(b)
# # # else:
# # #     print(b[0], b[1])

# # n = int(input())
# # res = []
# # for i in range(n):
# #     nums = list(map(int, input().split(' ')))
# #     res.append(nums)
# # # n = 10
# # dp = [1] * n
# # res1 = []
# # for j in range(n):
# #     nums = sorted([[res[j][0], res[j][1]], [res[j][2], res[j][3]], [res[j][4], res[j][5]]], key = lambda x : x[1])
# #     res1.append(nums)


# res1 = [[[1, 3], [2, 5], [4, 6]], [[3, 2], [5, 4], [1, 6]], [[6, 2], [1, 4], [3, 5]], [[4, 2], [6, 3], [1, 5]], [[2, 1], [5, 3], [6, 4]], [[2, 1], [4, 5], [3, 6]], [[3, 4], [2, 5], [1, 6]], [[5, 1], [4, 2], [6, 3]], [[3, 1], [6, 2], [5, 4]], [[6, 1], [4, 2], [5, 3]]];
# # for i in range(n):
# #     for j in range(i+1, n):
# #         ans = 0
# #         if res1[i][0] in res1[j]:
# #             ans += 1
# #         if res1[i][1] in res1[j]:
# #             ans += 1
# #         if res1[i][2] in res1[j]:
# #             ans += 1
# #         if reversed(res[i][0]) in res1[j] or reversed(res[i][1]) in res[j] or reversed(res[i][2]) in res1[j]:
# #             ans -= 1
# #         if ans >= 2:
# #             dp[i] += 1
# # dp.sort(reverse = True)
# # sezi = 0
# # for i in range(len(dp)):
# #     if sezi == n:
# #         break
# #     sezi += dp[i]
# # print(i)
# # for i in dp:
# #     if n == 0:
# #         break
# #     n -= i
# #     print(i, end = ' ')

# print(list(reversed(res1[0][0])))


n = int(input())
res = []
for i in range(n):
    nums = list(map(int, input().split(' ')))
    res.append(nums)
# n = 10
dp = [1] * n
res1 = []
for j in range(n):
    nums = sorted([[res[j][0], res[j][1]], [res[j][2], res[j][3]], [res[j][4], res[j][5]]], key = lambda x : x[1])
    res1.append(nums)


# res1 = [[[1, 3], [2, 5], [4, 6]], [[3, 2], [5, 4], [1, 6]], [[6, 2], [1, 4], [3, 5]], [[4, 2], [6, 3], [1, 5]], [[2, 1], [5, 3], [6, 4]], [[2, 1], [4, 5], [3, 6]], [[3, 4], [2, 5], [1, 6]], [[5, 1], [4, 2], [6, 3]], [[3, 1], [6, 2], [5, 4]], [[6, 1], [4, 2], [5, 3]]];
for i in range(n):
    for j in range(i+1, n):
        ans = 0
        if res1[i][0] in res1[j]:
            ans += 1
        if res1[i][1] in res1[j]:
            ans += 1
        if res1[i][2] in res1[j]:
            ans += 1
        if list(reversed(res1[i][0])) in res1[j] or list(reversed(res1[i][1])) in res[j] or list(reversed(res1[i][2])) in res1[j]:
            ans -= 1
        if ans >= 2:
            dp[i] += 1
dp.sort(reverse = True)
sezi = 0
for i in range(len(dp)):
    if sezi == n:
        break
    sezi += dp[i]
print(i)
for i in dp:
    if n == 0:
        break
    n -= i
    print(i, end = ' ')