# from heapq import heappop, heappush
# class Ugly:
#     def __init__(self):
#         seen = {1, }
#         self.nums = nums = []
#         heap = []
#         heappush(heap, 1)

#         for _ in range(1690):
#             curr_ugly = heappop(heap)
#             nums.append(curr_ugly)
#             for i in [2, 3, 5]:
#                 new_ugly = curr_ugly * i
#                 if new_ugly not in seen:
#                     seen.add(new_ugly)
#                     heappush(heap, new_ugly)
    
# class Solution:
#     u = Ugly()
#     def nthUglyNumber(self, n):
#         return self.u.nums[n - 1]

# a = Solution()
# b = a.nthUglyNumber(10)
# print(b)

# class Solution:
#     def singleNumber(self, nums):
#         seen_once = seen_twice = 0
        
#         for num in nums:
#             # first appearance: 
#             # add num to seen_once 
#             # don't add to seen_twice because of presence in seen_once
            
#             # second appearance: 
#             # remove num from seen_once 
#             # add num to seen_twice
            
#             # third appearance: 
#             # don't add to seen_once because of presence in seen_twice
#             # remove num from seen_twice
#             seen_once = ~seen_twice & (seen_once ^ num)
#             seen_twice = ~seen_once & (seen_twice ^ num)

#         return seen_once

# class Solution:
#     def singleNumber(self, nums):
#         res = 0
#         for i in range(32):
#             cnt = 0  # 记录当前 bit 有多少个1
#             bit = 1 << i  # 记录当前要操作的 bit
#             for num in nums:
#                 if num & bit != 0:
#                     cnt += 1
#             if cnt % 3 != 0:
#                 # 不等于0说明唯一出现的数字在这个 bit 上是1
#                 res |= bit

#         return res - 2 ** 32 if res > 2 ** 31 - 1 else res

# a = Solution()
# b = a.singleNumber([10,12,10,99,12,10,12])
# print(b)

# class Solution:
#     def findErrorNums(self, nums):
#         xor = xor0 = xor1 = 0
#         for n in nums:
#             xor ^= n
#         for i in range(1, len(nums) + 1):
#             xor ^= i
#         rightmostbit = xor & ~(xor - 1)
#         for n in nums:
#             if (n & rightmostbit) != 0:
#                 xor1 ^= n
#             else:
#                 xor0 ^= n
#         for i in range(1, len(nums) + 1):
#             if (i & rightmostbit) != 0:
#                 xor1 ^= i
#             else:
#                 xor0 ^= i
#         for i in range(len(nums)):
#             if nums[i] == xor0:
#                 return [xor0, xor1]
#         return [xor1, xor0]

# a = Solution()
# b = a.findErrorNums([1,2,2,4])
# print(b)

# class Solution:
#     def singleNumbers(self, nums):
#         ret = 0  # 所有数字异或的结果
#         a = 0
#         b = 0
#         for n in nums:
#             ret ^= n
#         # 找到第一位不是0的
#         h = 1
#         while(ret & h == 0):
#             h <<= 1
#         for n in nums:
#             # 根据该位是否为0将其分为两组
#             if (h & n == 0):
#                 a ^= n
#             else:
#                 b ^= n

#         return [a, b]

# a = Solution()
# b = a.singleNumbers([1,2,3,4,5,6,1,2,3,4])
# print(b)

# class Solution:
#     def reverseWords(self, s):
#         res = ""
#         s += " "
#         tmp = False
#         idx = 0
#         for i in range(len(s)):
#             if not tmp and s[i] != " ":
#                 idx = i
#                 tmp = True
#             elif tmp and s[i] == " ":
#                 tmp = False
#                 res = " " + s[idx:i] + res
#         return res[1:]

# class Solution:
#     def reverseWords(self, s: str) -> str:
#         s = s.strip()
#         res = []
#         i, j = len(s) - 1, len(s) - 1
#         while i >= 0:
#             while s[i] != ' ' and i >= 0: i -= 1
#             res.append(s[i+1:j+1])
#             while s[i] == ' ':  i -= 1
#             j = i
#         return ' '.join(res)

# class Solution:
#     def reverseWords(self, s):
#         s = s.strip()
#         left, right = 0, len(s) - 1
#         import collections            
#         d, word = collections.deque(), []
#         while left <= right:
#             if s[left] == ' ' and word:
#                 d.appendleft(''.join(word))
#                 word = []
#             elif s[left] != ' ':
#                 word.append(s[left])
#             left += 1
#         d.appendleft(''.join(word))
        
#         return ' '.join(d)

# a = Solution()
# b = a.reverseWords("   a good   example")
# print(b)

# from collections import Counter
# class Solution(object):
#     def threeSumMulti(self, A, target):
#         """
#         :type A: List[int]
#         :type target: int
#         :rtype: int
#         """
#         mod = 1000000007
        
#         counter = Counter(A)
        
#         ans = 0
#         # x != y != z
#         for x in range(101):
#             for y in range(x+1, 101):
#                 z = target - x - y
#                 if z <= y:
#                     continue
#                 ans += counter[x] * counter[y] * counter[z]
#                 ans %= mod 
                        
#         # x == y != z
#         for x in range(101):
#             z = target - 2*x
#             if z == x:
#                 continue
#             ans += counter[x] * (counter[x]-1) * counter[z] / 2
#             ans %= mod
            
#         if target % 3 == 0:
#             x = target / 3
#             ans += counter[x] * (counter[x]-1) * (counter[x]-2) / 6
#             ans %= mod
            
#         return int(ans)

# a = Solution()
# b = a.threeSumMulti([1,1,2,2,3,3,4,4,5,5], 8)
# print(b)

# class Solution:
#     def countDigitOne(self, n):
#         base, count, weight, roundd, temp = 1, 0, 0, 0, n
#         while temp > 0:
#             weight = temp % 10
#             roundd = temp // 10
#             temp = roundd
#             count += roundd * base
#             if weight > 1:
#                 count += base
#             elif weight == 1:
#                 count += n % base + 1
#             base = base * 10
#         return count

# a = Solution()
# b = a.countDigitOne(521)
# print(b)

# from collections import deque

# class Solution:
#     def updateMatrix(self, matrix):
#         m, n = len(matrix), len(matrix[0])
#         dist = [[0] * n for _ in range(m)]
#         zeroes_pos = [(i, j) for i in range(m) for j in range(n) if matrix[i][j] == 0]
#         # 将所有的 0 添加进初始队列中
#         q = deque(zeroes_pos)
#         seen = set(zeroes_pos)

#         # 广度优先搜索
#         while q:
#             i, j = q.popleft()
#             for ni, nj in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
#                 if 0 <= ni < m and 0 <= nj < n and (ni, nj) not in seen:
#                     dist[ni][nj] = dist[i][j] + 1
#                     q.append((ni, nj))
#                     seen.add((ni, nj))
        
#         return dist

# class Solution:
#     def updateMatrix(self, matrix):
#         m, n = len(matrix), len(matrix[0])
#         # 初始化动态规划的数组，所有的距离值都设置为一个很大的数
#         dist = [[float('inf')] * n for _ in range(m)]
#         # 如果 (i, j) 的元素为 0，那么距离为 0
#         for i in range(m):
#             for j in range(n):
#                 if matrix[i][j] == 0:
#                     dist[i][j] = 0
#         # 只有 水平向左移动 和 竖直向上移动，注意动态规划的计算顺序
#         for i in range(m):
#             for j in range(n):
#                 if i - 1 >= 0:
#                     dist[i][j] = min(dist[i][j], dist[i - 1][j] + 1)
#                 if j - 1 >= 0:
#                     dist[i][j] = min(dist[i][j], dist[i][j - 1] + 1)
#         # 只有 水平向右移动 和 竖直向下移动，注意动态规划的计算顺序
#         for i in range(m - 1, -1, -1):
#             for j in range(n - 1, -1, -1):
#                 if i + 1 < m:
#                     dist[i][j] = min(dist[i][j], dist[i + 1][j] + 1)
#                 if j + 1 < n:
#                     dist[i][j] = min(dist[i][j], dist[i][j + 1] + 1)
#         return dist

# a = Solution()
# b = a.updateMatrix([[0,0,0],[0,1,0],[1,1,1]])
# print(b)

# class Solution:
#     def mySqrt(self, x):
#         if x < 2:
#             return x
        
#         left, right = 2, x // 2
        
#         while left <= right:
#             pivot = left + (right - left) // 2
#             num = pivot * pivot
#             if num > x:
#                 right = pivot -1
#             elif num < x:
#                 left = pivot + 1
#             else:
#                 return pivot
            
#         return right

# a = Solution()
# b = a.mySqrt(10)
# print(b)

# class Solution:
#     def numIslands(self, grid):
#         res = 0
#         if not grid: return 0
#         for i in range(len(grid)):
#             for j in range(len(grid[0])):
#                 if grid[i][j] != 0:
#                     res += 1
#                     self.dfs(i, j, grid)
#         return res

#     def dfs(self, i, j, grid):
#         if 0 > i or i >= len(grid) or 0 > j or j >= len(grid[0]) or not grid[i][j]:
#             return 0
#         grid[i][j] = 0
#         return self.dfs(i+1, j, grid) + self.dfs(i-1, j, grid) + self.dfs(i, j+1, grid) + self.dfs(i, j-1, grid)

# a = Solution()
# b = a.numIslands([["1","1","0","0","0"],["1","1","0","0","0"],["0","0","1","0","0"],["0","0","0","1","1"]])
# print(b)

# class Solution:
#     def numberOfSubarrays(self, nums, k):
#         ans = 0
#         res = 0
#         dic = {0: 1}
#         for i in nums:
#             if i & 1:
#                 ans += 1
#                 dic[ans] = 1
#             else:
#                 dic[ans] += 1
#             res += dic.get(ans-k, 0)
#         return res

# a = Solution()
# b = a.numberOfSubarrays([1,2,2,2,1,2,2,1,2,2,2,1,2,2,2], 2)
# print(b)

# class Solution:
#     def longestCommonPrefix(self, strs):
#         if len(strs) == 0: return ""
#         prefix = strs[0]
#         for i in range(len(strs)):
#             while(strs[i].find(prefix) != 0):  # 不是其他字符串的前缀
#                 prefix = prefix[0 : len(prefix)-1]   # 减少前缀
#                 if not prefix:
#                     return ""
#         return prefix

# a = Solution()
# b = a.longestCommonPrefix(["flower","flow","flight"])
# print(b)

# for i in range(3):
#     for j in range(5):
#         print(j)
#         if j == 2:
#             break

# class Solution:
#     def permutation(self, s):
#         c, res = list(s), []
#         def dfs(x):
#             if x == len(c) - 1:
#                 res.append(''.join(c)) # 添加排列方案
#                 return
#             dic = set()
#             for i in range(x, len(c)):
#                 if c[i] in dic: continue # 重复，因此剪枝
#                 dic.add(c[i])
#                 c[i], c[x] = c[x], c[i] # 交换，将 c[i] 固定在第 x 位
#                 dfs(x + 1) # 开启固定第 x + 1 位字符
#                 c[i], c[x] = c[x], c[i] # 恢复交换
#         dfs(0)
#         return res

# a = Solution()
# b = a.permutation('abc')
# print(b)

# class Solution:
#     def waysToChange(self, n: int) -> int:
#         mod = 10**9 + 7
#         coins = [25, 10, 5, 1]

#         f = [1] + [0] * n
#         for coin in coins:
#             for i in range(coin, n + 1):
#                 f[i] += f[i - coin]
#         return f[n] % mod

# a = Solution()
# b = a.waysToChange(10)
# print(b)

# class Solution:
#     def firstUniqChar(self, s: str) -> str:
#         dic = {}
#         for c in s:
#             dic[c] = not c in dic
#         for k, v in dic.items():
#             if v: return k
#         return ' '

# a = Solution()
# b = a.firstUniqChar('leetcode')
# print(b)

# class Solution:
#     def reversePairs(self, nums):
#         self.cnt = 0
#         def merge(nums, start, mid, end, temp):
#             i, j = start, mid + 1
#             while i <= mid and j <= end:
#                 if nums[i] <= nums[j]:
#                     temp.append(nums[i])
#                     i += 1
#                 else:
#                     self.cnt += mid - i + 1
#                     temp.append(nums[j])
#                     j += 1
#             while i <= mid:
#                 temp.append(nums[i])
#                 i += 1
#             while j <= end:
#                 temp.append(nums[j])
#                 j += 1
            
#             for i in range(len(temp)):
#                 nums[start + i] = temp[i]
#             temp.clear()
                    

#         def mergeSort(nums, start, end, temp):
#             if start >= end: return
#             mid = (start + end) >> 1
#             mergeSort(nums, start, mid, temp)
#             mergeSort(nums, mid + 1, end, temp)
#             merge(nums, start, mid,  end, temp)
#         mergeSort(nums, 0, len(nums) - 1, [])
#         return self.cnt

# class Solution:
#     def reversePairs(self, nums):
#         def Merge(nums, left, mid, right, temp):
#             temp = nums[mid + 1: right + 1]
#             i, j, k = mid, right - mid - 1, right
#             while i >= left and j >= 0:
#                 if nums[i] > temp[j]:
#                     nums[k] = nums[i]
#                     self.cnt += j + 1
#                     i -= 1
#                 else:
#                     nums[k] = temp[j]
#                     j -= 1
#                 k -= 1
#             if j >= 0: nums[left:k+1] = temp[:j+1]

#         def MSort(nums, left, right, temp):
#             if left < right:
#                 mid = left + (right - left) // 2
#                 MSort(nums, left, mid, temp)
#                 MSort(nums, mid + 1, right, temp)
#                 if nums[mid] <= nums[mid + 1]: return 
#                 Merge(nums, left, mid, right, temp)
                
#         self.cnt = 0
#         MSort(nums, 0, len(nums) - 1, [])
#         return self.cnt

# class Solution(object):
#     def reversePairs(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: int
#         """
#         result = [0]
#         def Merge_sort(s):
#             n = len(s)
#             if n < 2:
#                 return
#             mid = n // 2
#             s1 = s[0:mid]
#             s2 = s[mid:n]
#             Merge_sort(s1)
#             Merge_sort(s2)
#             Merge(s1,s2,s)

#         def Merge(s1,s2,s):
#             len_s1 = len(s1) - 1
#             len_s2 = len(s2) - 1
#             temp = len(s) - 1
            
#             while len_s1 >=0 and len_s2 >= 0:
#                 if s1[len_s1] > s2[len_s2]:
#                     s[temp] = s1[len_s1]
#                     result[0] += len_s2 + 1
#                     len_s1 -= 1
#                     temp -= 1
#                 else:
#                     s[temp] = s2[len_s2]
#                     len_s2 -= 1
#                     temp -= 1
                    
#             while len_s1 >= 0:
#                 s[temp] = s1[len_s1]
#                 len_s1 -= 1
#                 temp -= 1
#             while len_s2 >= 0:
#                 s[temp] = s2[len_s2]
#                 temp -= 1
#                 len_s2 -= 1

#         Merge_sort(nums)
#         return result[0]

# a = Solution()
# b = a.reversePairs([7,5,6,4])
# print(b)

# from functools import lru_cache
# class Solution:
#     def mincostTickets(self, days, costs):
#         N = len(days)
#         durations = [1, 7, 30]

#         @lru_cache(None)
#         def dp(i):
#             if i >= N:
#                 return 0
#             ans = 10**9
#             j = i
#             for c, d in zip(costs, durations):
#                 while j < N and days[j] < days[i] + d:
#                     j += 1
#                 ans = min(ans, dp(j) + c)
#             return ans

#         return dp(0)

# a = Solution()
# b = a.mincostTickets([1,2,3,4,5,6,7,8,9,10,30,31], [2,7,15])
# print(b)

# import collections

# class Solution:
#     def subarraySum(self, nums, k):
#         # num_times 存储某“前缀和”出现的次数，这里用collections.defaultdict来定义它
#         # 如果某前缀不在此字典中，那么它对应的次数为0
#         num_times = collections.defaultdict(int)
#         num_times[0] = 1  # 先给定一个初始值，代表前缀和为0的出现了一次
#         cur_sum = 0  # 记录到当前位置的前缀和
#         res = 0
#         for i in range(len(nums)):
#             cur_sum += nums[i]  # 计算当前前缀和
#             if cur_sum - k in num_times:  # 如果前缀和减去目标值k所得到的值在字典中出现，即当前位置前缀和减去之前某一位的前缀和等于目标值
#                 res += num_times[cur_sum - k]
#             # 下面一句实际上对应两种情况，一种是某cur_sum之前出现过（直接在原来出现的次数上+1即可），
#             # 另一种是某cur_sum没出现过（理论上应该设为1，但是因为此处用defaultdict存储，如果cur_sum这个key不存在将返回默认的int，也就是0）
#             # 返回0加上1和直接将其置为1是一样的效果。所以这里统一用一句话包含上述两种情况
#             num_times[cur_sum] += 1
#         return res

# a = Solution()
# b = a.subarraySum([1,1,0,1,0,2], 2)
# print(b)

# import collections

# class Solution:
#     def findOrder(self, numCourses, prerequisites):
#         # 存储有向图
#         edges = collections.defaultdict(list)
#         # 存储每个节点的入度
#         indeg = [0] * numCourses
#         # 存储答案
#         result = list()

#         for info in prerequisites:
#             edges[info[1]].append(info[0])
#             indeg[info[0]] += 1
        
#         # 将所有入度为 0 的节点放入队列中
#         q = collections.deque([u for u in range(numCourses) if indeg[u] == 0])

#         while q:
#             # 从队首取出一个节点
#             u = q.popleft()
#             # 放入答案中
#             result.append(u)
#             for v in edges[u]:
#                 indeg[v] -= 1
#                 # 如果相邻节点 v 的入度为 0，就可以选 v 对应的课程了
#                 if indeg[v] == 0:
#                     q.append(v)

#         if len(result) != numCourses:
#             result = list()
#         return result

# a = Solution()
# b = a.findOrder(4, [[1,0],[2,0],[3,1],[3,2]])
# print(b)

# class Solution:
#     def maxProduct(self, nums):
#         A = nums
#         B = A[::-1]
#         for i in range(1, len(A)):
#             A[i] *= A[i-1] or 1
#             B[i] *= B[i-1] or 1
#         return max(max(A), max(B))

# a = Solution()
# b = a.maxProduct([2,3,0,-1,-2,3,6])
# print(b)

# class Solution:
#     def findTheLongestSubstring(self, s: str) -> int:
#         D = {"a": 1, "e": 2, "i": 4, "o": 8, "u": 16}
#         L = {0: 0}
#         m = t = 0
#         for i, c in enumerate(s, 1):
#             t ^= D.get(c, 0)
#             m = max(m, i - L.setdefault(t, i))
#         return m

# a = Solution()
# b = da.findTheLongestSubstring('leetcodevscode')
# print(b)

# class Solution:
#     def longestPalindrome(self, s: str) -> str:
#         n = len(s)
#         if n < 2 or s == s[::-1]:
#             return s
#         max_len = 1
#         start = 0
#         for i in range(1,n):
#             even = s[i-max_len:i+1]
#             odd = s[i-max_len-1:i+1]
#             if i-max_len-1>=0 and odd == odd[::-1]:
#                 start = i-max_len-1
#                 max_len += 2
#                 continue
#             if i-max_len>=0 and even == even[::-1]:
#                 start = i-max_len
#                 max_len += 1
#         return s[start:start+max_len]

# a = Solution()
# b = a.longestPalindrome('abccb')
# print(b)

# class Solution:
#     def largestRectangleArea(self, heights):
#         ans, s, hs = 0, [0], [0, *heights, 0]
#         for i, h in enumerate(hs):
#             while hs[s[-1]] > h:
#                 ans = max(ans, (i - s[-2] - 1) * hs[s.pop()])
#             s.append(i)
#         return ans

# a = Solution()
# b = a.largestRectangleArea([2,1,5,6,2,3])
# print(b)

# class Solution:
#     def wordBreak(self, s, wordDict):
#         wordDict=set(wordDict)
#         visited={}
#         def dfs(s,wordDict):
#             res=[]
#             if s in visited:
#                 return visited[s]
#             if not s:
#                 return []
#             lenth=len(s)
#             for i in wordDict:
#                 if not s.startswith(i):
#                     continue
#                 if lenth==len(i):
#                     res.append(i)
#                 temp=dfs(s[len(i):],wordDict)
#                 for j in temp:
#                     res.append(i+' '+j)
#             visited[s]=res
#             return res
#         return dfs(s,wordDict)

# class Solution:
#     def wordBreak(self, s, wordDict):
#         tmp = set("".join(wordDict))
#         if any([i not in tmp for i in s]):
#             return []
#         dp = [[""], [s[0]]*(s[0] in wordDict)]
#         for i in range(1, len(s)):
#             dp.append(sum([[f"{k} {s[j: i+1]}" if k else s[j: i+1] for k in dp[j]] for j in range(i+1) if s[j: i+1] in wordDict and dp[j]], []))
#         return dp[-1]

# a = Solution()
# b = a.wordBreak("catsanddog",["cat","cats","and","sand","dog"])
# print(b)

# class Solution:
#     def kthSmallest(self, matrix, k):
#         n = len(matrix)

#         def check(mid):
#             i, j = n - 1, 0
#             num = 0
#             while i >= 0 and j < n:
#                 if matrix[i][j] <= mid:
#                     num += i + 1
#                     j += 1
#                 else:
#                     i -= 1
#             return num >= k

#         left, right = matrix[0][0], matrix[-1][-1]
#         while left < right:
#             mid = (left + right) // 2
#             if check(mid):
#                 right = mid
#             else:
#                 left = mid + 1
        
#         return left

# a = Solution()
# b = a.kthSmallest([[1,5,9],[10,11,13],[12,13,15]],8)
# print(b)

# class Solution:
#     def longestValidParentheses(self, s: str) -> int:
#         stack = [-1]
#         length = 0
#         max_length = 0
#         for i in range(len(s)):
#             if s[i] == '(':
#                 stack.append(i)
#             else:
#                 stack.pop()
#                 if stack == []:
#                     stack.append(i)
#                 else:
#                     length = i-stack[-1]
#                     max_length = max(max_length,length)
#         return max_length

# a = Solution()
# b = a.longestValidParentheses('(())))(())()(')
# print(b)

# class Solution:
#     def isValidSerialization(self, preorder: str) -> bool:
#         # number of available slots
#         slots = 1

#         for node in preorder.split(','):
#             # one node takes one slot
#             slots -= 1
            
#             # no more slots available
#             if slots < 0:
#                 return False
            
#             # non-empty node creates two children slots
#             if node != '#':
#                 slots += 2
        
#         # all slots should be used up
#         return slots == 0

# class Solution:
#     def isValidSerialization(self, preorder: str) -> bool:
#         # number of available slots
#         slots = 1
        
#         prev = None  # previous character
#         for ch in preorder:
#             if ch == ',':
#                 # one node takes one slot
#                 slots -= 1

#                 # no more slots available
#                 if slots < 0:
#                     return False

#                 # non-empty node creates two children slots
#                 if prev != '#':
#                     slots += 2
#             prev = ch
        
#         # the last node
#         slots = slots + 1 if ch != '#' else slots - 1 
#         # all slots should be used up
#         return slots == 0

# a = Solution()
# b = a.isValidSerialization("9,3,4,#,#,1,#,#,2,#,6,#,#")
# print(b)

# class Solution:
#     def calculateMinimumHP(self, dungeon):
#         n, m = len(dungeon), len(dungeon[0])
#         BIG = 10 ** 9
#         dp = [[BIG] * (m + 1) for _ in range(n + 1)]
#         dp[n-1][m] = dp[n][m-1] = 1
#         for i in range(n-1, -1, -1):
#             for j in range(m-1, -1, -1):
#                 minn = min(dp[i+1][j], dp[i][j+1])
#                 dp[i][j] = max(minn-dungeon[i][j], 1)
#         return dp[0][0]   

# a = Solution()
# b = a.calculateMinimumHP([[-2,-3,3],[-5,-10,1],[10,30,-5]])
# print(b)

# class Solution:
#     def reverseLeftWords(self, s: str, n: int) -> str:
#         res = []
#         for i in range(n, n + len(s)):
#             res.append(s[i % len(s)])
#         return ''.join(res)

# a = Solution()
# b = a.reverseLeftWords('abcdefg', 3)
# print(b)

# class Solution:
#     def __init__(self):
#         self.res = 0
#     def sumNums(self, n: int) -> int:
#         n > 1 and self.sumNums(n-1)
#         self.res += n
#         return self.res

# a = Solution()
# b = a.sumNums(-10)
# print(b)

# class Solution(object):
#     def generateParenthesis(self, N):
#         ans = []
#         def backtrack(S = '', left = 0, right = 0):
#             if len(S) == 2 * N:
#                 ans.append(S)
#                 return
#             if left < N:
#                 backtrack(S+'(', left+1, right)
#             if right < left:
#                 backtrack(S+')', left, right+1)

#         backtrack()
#         return ans

# a = Solution()
# b = a.generateParenthesis(3)
# print(b)

# class Solution:
#     def splitArray(self, nums, m):
#         def check(x):
#             total, cnt = 0, 1
#             for num in nums:
#                 if total + num > x:
#                     cnt += 1
#                     total = num
#                 else:
#                     total += num
#             return cnt <= m


#         left = max(nums)
#         right = sum(nums)
#         while left < right:
#             mid = (left + right) // 2
#             if check(mid):
#                 right = mid
#             else:
#                 left = mid + 1

#         return left

# a = Solution()
# b = a.splitArray([7,2,5,10,8], 2)
# print(b)

# class Solution(object):
#     def longestIncreasingPath(self, matrix):
#         if not matrix or not matrix[0]:
#             return 0
#         m, n = len(matrix), len(matrix[0])
#         lst = []
#         for i in range(m):
#             for j in range(n):
#                 lst.append((matrix[i][j], i, j))
#         lst.sort()
#         dp = [[0 for _ in range(n)] for _ in range(m)]
#         for num, i, j in lst:
#             dp[i][j] = 1
#             for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
#                 r, c = i + di, j + dj
#                 if 0 <= r < m and 0 <= c < n:
#                     if matrix[i][j] > matrix[r][c]:
#                         dp[i][j] = max(dp[i][j], 1 + dp[r][c])
#         return max([dp[i][j] for i in range(m) for j in range(n)])

# a = Solution()
# b = a.longestIncreasingPath([[9,9,4],[6,6,8],[2,1,1]])
# print(b)

# import bisect
# class Solution:
    
#     def maxValue(self, meeting, value):
        
#         mv = sorted(list((*m, v) for m, v in zip(meeting, value)), key = lambda x:x[1])
    
#         dp = [0] + [0] * len(value)
#         timeline = [0]
        
#         for j, (start, end, value) in enumerate(mv, start=1):
#             prev = bisect.bisect(timeline, start) - 1
#             dp[j] = max(dp[j-1], dp[prev]+value)
#             timeline.append(end)
#         return dp[-1]

# a = Solution()
# b = a.maxValue([[10,40],[20,50],[30,45],[40,60]], [3, 6, 2, 4])
# print(b)

# class Solution():
#     def minMeetingRooms(self, intervals):
#         occupied, res = [], 0
#         intervals = sorted(intervals, key = lambda x : x[0])
#         for i in intervals:
#             start, end = i
#             occupied = [t for t in occupied if t > start]
#             occupied.append(end)
#             res = max(res, len(occupied))
#         return res

# a = Solution()
# b = a.minMeetingRooms([[0, 30],[5, 10],[15, 20],[25, 30],[15, 25], [5, 15]])
# print(b)

# class Solution:
#     def canFinish(self, numCourses, prerequisites):
#         def dfs(i, adjacency, flags):
#             if flags[i] == -1: return True
#             if flags[i] == 1: return False
#             flags[i] = 1
#             for j in adjacency[i]:
#                 if not dfs(j, adjacency, flags): return False
#             flags[i] = -1
#             return True

#         adjacency = [[] for _ in range(numCourses)]
#         flags = [0 for _ in range(numCourses)]
#         for cur, pre in prerequisites:
#             adjacency[pre].append(cur)
#         for i in range(numCourses):
#             if not dfs(i, adjacency, flags): return False
#         return True

# from collections import deque

# class Solution:
#     def canFinish(self, numCourses, prerequisites):
#         indegrees = [0 for _ in range(numCourses)]
#         adjacency = [[] for _ in range(numCourses)]
#         queue = deque()
#         # Get the indegree and adjacency of every course.
#         for cur, pre in prerequisites:
#             indegrees[cur] += 1
#             adjacency[pre].append(cur)
#         # Get all the courses with the indegree of 0.
#         for i in range(len(indegrees)):
#             if not indegrees[i]: queue.append(i)
#         # BFS TopSort.
#         while queue:
#             pre = queue.popleft()
#             numCourses -= 1
#             for cur in adjacency[pre]:
#                 indegrees[cur] -= 1
#                 if not indegrees[cur]: queue.append(cur)
#         return not numCourses

# a = Solution()
# b = a.canFinish(6, [[1,2],[2,4],[2,3],[4,5],[3,5],[1,4]])
# print(b)

# class Solution(object):
#     def restoreIpAddresses(self, s):
#         """
#         :type s: str
#         :rtype: List[str]
#         """
#         self.res = []

#         def backtrack(s, tmp):
#             if len(s) == 0 and len(tmp) == 4:
#                 self.res.append('.'.join(tmp))
#                 return
#             if len(tmp) < 4:
#                 for i in range(min(3, len(s))):
#                     p, n = s[:i + 1], s[i + 1:]
#                     if p and 0 <= int(p) <= 255 and str(int(p)) == p:
#                         backtrack(n, tmp + [p])

#         backtrack(s, [])
#         return self.res

# a = Solution()
# b = a.restoreIpAddresses("25525511135")
# print(b)

# class Solution:
#     def solve(self, board):
#         if not board:
#             return
        
#         n, m = len(board), len(board[0])

#         def dfs(x, y):
#             if not 0 <= x < n or not 0 <= y < m or board[x][y] != 'O':
#                 return
            
#             board[x][y] = "A"
#             dfs(x + 1, y)
#             dfs(x - 1, y)
#             dfs(x, y + 1)
#             dfs(x, y - 1)
        
#         for i in range(n):
#             dfs(i, 0)
#             dfs(i, m - 1)
        
#         for i in range(1, m - 1):
#             dfs(0, i)
#             dfs(n - 1, i)
        
#         for i in range(n):
#             for j in range(m):
#                 if board[i][j] == "A":
#                     board[i][j] = "O"
#                 elif board[i][j] == "O":
#                     board[i][j] = "X"
#         return board

# a = Solution()
# b = a.solve([["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]])
# print(b)

# class Solution:
#     def floodFill(self, image, sr, sc, newColor):
#         color = image[sr][sc]
#         if color == newColor:
#             return image
#         def dfs(i, j):
#             if 0 > i  or i >= len(image) or  0 > j or j >= len(image[0]) or image[i][j] != color:
#                 return
#             image[i][j] = newColor
#             dfs(i+1, j)
#             dfs(i-1, j)
#             dfs(i, j+1)
#             dfs(i, j-1)

#         dfs(sr, sc)

#         return image
# a = Solution()
# b = a.floodFill([[1,1,1],[1,1,0],[1,0,1]], 1, 1, 2)
# print(b)

# class Solution:
#     def insertion_sort(self, nums):
#         # 第一层for表示循环插入的遍数
#         for i in range(1, len(nums)):
#             for j in range(i, 0,  -1):
#                 if nums[j] < nums[j-1]:
#                     nums[j], nums[j-1] = nums[j-1], nums[j]
#                 else:
#                     break
#         return nums

# a = Solution()
# b = a.insertion_sort([7, 6, 4, 5, 3, 2, 1])
# print(b)

# class Solution:
#     def repeatedSubstringPattern(self, s):
#         n = len(s)
#         for i in range(1, n // 2 + 1):
#             if n % i == 0:
#                 if all(s[j] == s[j - i] for j in range(i, n)):
#                     return True
#         return False

# a = Solution()
# b = a.repeatedSubstringPattern('abcabcabc')
# print(b)

# import collections

# class Solution:
#     def findItinerary(self, tickets):
#         paths = collections.defaultdict(list)
#         for start, tar in tickets:
#             paths[start].append(tar)
#         for start in paths:
#             paths[start].sort(reverse=True)
#         s = []

#         def search(start):
#             while paths[start]:
#                 search(paths[start].pop())
#             s.append(start)

#         search("JFK")
#         return s[::-1]

# a = Solution()
# b = a.findItinerary([["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]])
# print(b)


# num = [5, 3, 2]
# n, k, d = num[0], num[1], num[2]
# nums = list(range(1, k+1))

# class Solution():
#     def helper(self, nums, target, d):
#         path = []
#         size = len(nums) - 1
#         res = []
#         dic = {}
#         def dfs(nums, begin, size, path, res, target, dic):
#             if target == 0:
#                 if max(path) >= d:
#                     res.append(path[:])
#                     return 1
#             elif target < nums[0]:
#                 return 0
#             count = 0
#             for index in range(size, begin-1, -1):
#                 resduie = target - nums[index]
# #                 if resduie < 0:
# #                     break
#                 if resduie not in dic:
#                     path.append(nums[index])
#                     cur = dfs(nums, begin, size, path, res, resduie, dic)
#                     dic[resduie] = cur
#                     count += cur
#                     path.pop()
#                 else:
#                     count += dic[resduie]
#             return count
#         count = dfs(nums, 0, size, path, res, target, dic)
#         return count

# a = Solution()
# b = a.helper(nums, n, d)
# print(b%998244353)

import functools
class Solution:
    def combinationSum4(self, nums, target):
        n = len(nums)
        nums.sort()
        @functools.lru_cache(None)
        def helper(res):
            if res == target:
                return 1
            ans = 0
            for i in range(n):
                val = res + nums[i]
                if val > target:
                    break
                ans += helper(val)
            return ans
        return helper(0)

a = Solution()
b = a.combinationSum4([1,2,3], 5)
print(b)