# import functools

# class Solution:
#     def coinChange(self, coins, amount):
#         @functools.lru_cache(amount)
#         def dp(rem):
#             if rem < 0: return -1
#             if rem == 0: return 0
#             mini = int(1e9)
#             for coin in self.coins:
#                 res = dp(rem - coin)
#                 if res >= 0 and res < mini:
#                     mini = res + 1
#             return mini if mini < int(1e9) else -1

#         self.coins = coins
#         if amount < 1: return 0
#         return dp(amount)

# 零钱兑换
# class Solution:
#     def coinChange(self, coins, amount):
#         dp = [float('inf')] * (amount + 1)
#         dp[0] = 0
        
#         for coin in coins:
#             for x in range(coin, amount + 1):
#                 dp[x] = min(dp[x], dp[x - coin] + 1)
#         return dp[amount] if dp[amount] != float('inf') else -1

# a = Solution()
# b = a.coinChange([1,2,5],11)
# print(b)


# import collections
# class Solution:
#     def groupAnagrams(self, strs):
#         dic = collections.defaultdict(list)
#         for s in strs : dic["".join(sorted(s))].append(s)
#         return list(dic.values())

# a = Solution()
# b = a.groupAnagrams(["eat","tea","tan","ate","nat","bat"])
# print(b)

# class Solution:
#     def maxProfit(self, prices):
#         last = 0
#         profit = 0
#         for i in range(len(prices)-1):
#             last = max(0, last + prices[i+1] - prices[i])
#             profit = max(profit, last)
#         return profit

# a = Solution()
# b = a.maxProfit([7,1,6,3,6,4,7,5,9])
# print(b)

# class Solution:
#     def countDigitOne(self, n):
#         if n<=0:return 0
#         elif n<10:return 1
#         last=int(str(n)[1:])
#         weight=10**(len(str(n))-1)
#         high=int(str(n)[0])
#         if high==1:
#             """
#             例如：135中1的次数：high=1,last=35,weight=100
#             100-135百位1出现共last+1，00-35共countDigitOne(last-1)
#             0-99共countDigitOne(weight-1)
#             """
#             return self.countDigitOne(last)+last+1+self.countDigitOne(weight-1)
#         else:
#             """
#             例如：535中1的次数：high=5>1,last=35,weight=100
#             100-199百位1出现共weight
#             0-99,100-199,200-299...400-499非百位1共出现high*countDigitOne(weight-1)
#             500-535出现1次数countDigitOne(last)
#             """
#             return high*self.countDigitOne(weight-1)+self.countDigitOne(last)+weight

# a = Solution()
# b = a.countDigitOne(100)
# print(b)

# class Solution:
#     def findNthDigit(self, n):
#         n -= 1
#         for digits in range(1, 11):
#             first_num = 10**(digits - 1)
#             if n < 9 * first_num * digits:
#                 return int(str(first_num + n/digits)[n%digits])
#             n -= 9 * first_num * digits

# a = Solution()
# b = a.findNthDigit(363)
# print(b)

# class Solution:
#     def lengthOfLongestSubstring(self, s: str) -> int:
#         max = 0
#         i = 0
#         for j in range(len(s)):
#             for k in range(i,j):
#                 if(s[k] == s[j]):
#                     i = k + 1
#                     break
#             if(j-i+1 > max):
#                 max = j-i+1
#         return max

# a = Solution()
# b = a.lengthOfLongestSubstring('abcabcdef')
# print(b)

# class Solution:
#     def maxSubArray(self, nums):
#         n = len(nums)
#         max_sum = nums[0]
#         for i in range(1, n):
#             if nums[i - 1] > 0:
#                 nums[i] += nums[i - 1] 
#             max_sum = max(nums[i], max_sum)

#         return max_sum

# class Solution:
#     def maxSubArray(self, nums: 'List[int]') -> 'int':
#         n = len(nums)
#         curr_sum = max_sum = nums[0]

#         for i in range(1, n):
#             curr_sum = max(nums[i], curr_sum + nums[i])
#             max_sum = max(max_sum, curr_sum)
            
#         return max_sum

# class Solution:
#     def cross_sum(self, nums, left, right, p): 
#             if left == right:
#                 return nums[left]

#             left_subsum = float('-inf')
#             curr_sum = 0
#             for i in range(p, left - 1, -1):
#                 curr_sum += nums[i]
#                 left_subsum = max(left_subsum, curr_sum)

#             right_subsum = float('-inf')
#             curr_sum = 0
#             for i in range(p + 1, right + 1):
#                 curr_sum += nums[i]
#                 right_subsum = max(right_subsum, curr_sum)

#             return left_subsum + right_subsum   
    
#     def helper(self, nums, left, right): 
#         if left == right:
#             return nums[left]
        
#         p = (left + right) // 2
            
#         left_sum = self.helper(nums, left, p)
#         right_sum = self.helper(nums, p + 1, right)
#         cross_sum = self.cross_sum(nums, left, right, p)
        
#         return max(left_sum, right_sum, cross_sum)
        
#     def maxSubArray(self, nums: 'List[int]') -> 'int':
#         return self.helper(nums, 0, len(nums) - 1)


# a = Solution()
# b = a.maxSubArray([-2,1,-3,4,-1,2,1,-5,4])
# print(b)

# class Solution:
#     def gcdOfStrings(self, str1, str2):
#         for i in range(min(len(str1), len(str2)), 0, -1):
#             if (len(str1) % i) == 0 and (len(str2) % i) == 0:
#                 if str1[: i] * (len(str1) // i) == str1 and str1[: i] * (len(str2) // i) == str2:
#                     return str1[: i]
#         return ''

# import math

# class Solution:
#     def gcdOfStrings(self, str1, str2):
#         candidate_len = math.gcd(len(str1), len(str2))
#         candidate = str1[: candidate_len]
#         if candidate * (len(str1) // candidate_len) == str1 and candidate * (len(str2) // candidate_len) == str2:
#             return candidate
#         return ''

# import math

# class Solution:
#     def gcdOfStrings(self, str1, str2):
#         candidate_len = math.gcd(len(str1), len(str2))
#         candidate = str1[: candidate_len]
#         if str1 + str2 == str2 + str1:
#             return candidate
#         return ''

# a = Solution()
# b = a.gcdOfStrings('ABABAB','AB')
# print(b)

# from typing import List


# class Solution:
#     #         (x-1,y)
#     # (x,y-1) (x,y) (x,y+1)
#     #         (x+1,y)

#     directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]

#     def exist(self, board: List[List[str]], word: str) -> bool:
#         m = len(board)
#         if m == 0:
#             return False
#         n = len(board[0])

#         marked = [[False for _ in range(n)] for _ in range(m)]
#         for i in range(m):
#             for j in range(n):
#                 # 对每一个格子都从头开始搜索
#                 if self.__search_word(board, word, 0, i, j, marked, m, n):
#                     return True
#         return False

#     def __search_word(self, board, word, index,start_x, start_y, marked, m, n):
#         # 先写递归终止条件
#         if index == len(word) - 1:
#             return board[start_x][start_y] == word[index]

#         # 中间匹配了，再继续搜索
#         if board[start_x][start_y] == word[index]:
#             # 先占住这个位置，搜索不成功的话，要释放掉
#             marked[start_x][start_y] = True
#             for direction in self.directions:
#                 new_x = start_x + direction[0]
#                 new_y = start_y + direction[1]
#                 # 注意：如果这一次 search word 成功的话，就返回
#                 if 0 <= new_x < m and 0 <= new_y < n and \
#                         not marked[new_x][new_y] and \
#                         self.__search_word(board, word,index + 1,new_x, new_y,marked, m, n):
#                     return True
#             marked[start_x][start_y] = False
#         return False

# a = Solution()
# b = a.exist([["A","B","C","E"],["S","F","C","S"],["A","D","C","E"]],"SFCCD")
# print(b)

# while 1:
#     nm = input()
#     if nm != '':
#         n,m = map(int,nm.split())
#         price = list(map(int,input().split()))
#         price = sorted(price)
#         buylist = {}
#         for i in range(m):
#             cargo = input()
#             buylist.setdefault(cargo,0)
#             buylist[cargo]+=1
#         count = sorted(buylist.values(),reverse = True)
#         minsum=0
#         maxsum=0
#         for i,num in enumerate(count):
#             maxsum += num*price[(-i-1)]
#             minsum += num*price[i]
#         print(minsum,maxsum)

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# def generaList(l):
#     prenode = ListNode(0)
#     lastnode = prenode
#     for val in l:
#         lastnode.next = ListNode(val)
#         lastnode = lastnode.next
#     return prenode.next
# def printList(l):
#     while l:
#         print(l.val, end='')
#         l = l.next

# class Solution:
#     def reverseList(self, head):
#         prev = None
#         curr = generaList([1,2,3,4,5])
#         while curr != None:
#             nextTemp = curr.next
#             print(curr.val)
#             curr.next = prev
#             prev = curr
#             curr = nextTemp
#         return prev

# a = Solution()
# b = a.reverseList('a')
# printList(b)

# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# def generateList(l):
#     prenode = ListNode(0)
#     lastnode = prenode
#     for val in l:
#         lastnode.next = ListNode(val)
#         lastnode = lastnode.next
#     return prenode.next
# def printList(l):
#     while l:
#         print(l.val,end='')
#         l = l.next
# l1 = generateList([1,2,3,4,5])
# printList(l1)
# class Solution:
#     def reverseList(self, head):
#         if head == None or head.next == None: return head
#         p = self.reverseList(head.next)
#         head.next.next = head
#         head.next = None
#         return p
# a = Solution()
# b = a.reverseList(l1)
# print('')
# printList(b)

# class Solution(object):
#     def spiralOrder(self, matrix):
#         if not matrix: return []
#         R, C = len(matrix), len(matrix[0])
#         seen = [[False] * C for _ in matrix]
#         ans = []
#         dr = [0, 1, 0, -1]
#         dc = [1, 0, -1, 0]
#         r = c = di = 0
#         for _ in range(R * C):
#             ans.append(matrix[r][c])
#             seen[r][c] = True
#             cr, cc = r + dr[di], c + dc[di]
#             if 0 <= cr < R and 0 <= cc < C and not seen[cr][cc]:
#                 r, c = cr, cc
#             else:
#                 di = (di + 1) % 4
#                 r, c = r + dr[di], c + dc[di]
#         return ans

# class Solution(object):
#     def spiralOrder(self, matrix):
#         def spiral_coords(r1, c1, r2, c2):
#             for c in range(c1, c2 + 1):
#                 yield r1, c
#             for r in range(r1 + 1, r2 + 1):
#                 yield r, c2
#             if r1 < r2 and c1 < c2:
#                 for c in range(c2 - 1, c1, -1):
#                     yield r2, c
#                 for r in range(r2, r1, -1):
#                     yield r, c1

#         if not matrix: return []
#         ans = []
#         r1, r2 = 0, len(matrix) - 1
#         c1, c2 = 0, len(matrix[0]) - 1
#         while r1 <= r2 and c1 <= c2:
#             for r, c in spiral_coords(r1, c1, r2, c2):
#                 ans.append(matrix[r][c])
#             r1 += 1; r2 -= 1
#             c1 += 1; c2 -= 1
#         return ans

# a = Solution()
# b = a.spiralOrder([[1,2,3],[4,5,6],[7,8,9]])
# print(b)

# class Solution:
#     def lengthOfLIS(self, nums):
#         if not nums:
#             return 0
#         dp = []
#         for i in range(len(nums)):
#             dp.append(1)
#             for j in range(i):
#                 if nums[i] > nums[j]:
#                     dp[i] = max(dp[i], dp[j] + 1)
#         return max(dp)

# class Solution:
#     def lengthOfLIS(self, nums):
#         d = []
#         for n in nums:
#             if not d or n > d[-1]:
#                 d.append(n)
#             else:
#                 l, r = 0, len(d) - 1
#                 loc = r
#                 while l <= r:
#                     mid = (l + r) // 2
#                     if d[mid] >= n:
#                         loc = mid
#                         r = mid - 1
#                     else:
#                         l = mid + 1
#                 d[loc] = n
#         return len(d)

# Dynamic programming + Dichotomy.
# class Solution:
#     def lengthOfLIS(self, nums: [int]) -> int:
#         tails, res = [0] * len(nums), 0
#         for num in nums:
#             i, j = 0, res
#             while i < j:
#                 m = (i + j) // 2
#                 if tails[m] < num: i = m + 1 # 如果要求非严格递增，将此行 '<' 改为 '<=' 即可。
#                 else: j = m
#             tails[i] = num
#             if j == res: res += 1
#         return res

# a = Solution()
# b = a.lengthOfLIS([10,9,2,5,3,7,101,18])
# print(b)

# class Solution:
#     def canJump(self, nums):
#         start = 0
#         end = 0
#         n = len(nums)
#         while start <= end and end < len(nums) - 1:
#             end = max(end, nums[start] + start)
#             start += 1
#         return end >= n - 1

# a = Solution()
# b = a.canJump([2,3,1,1,0,4])
# print(b)

# class Solution:
#     def maxAreaOfIsland(self, grid):
#         """可以用dfs做，也可以用并查集做，还是dfs吧，思路清晰一些"""
#         self.grid = grid
#         self.m = len(grid)
#         self.n = len(grid[0])
#         res = 0

#         for i in range(self.m):
#             for j in range(self.n):
#                 if self.grid[i][j] == 1:
#                     res = max(res, self._dfs(i, j))

#         return res

#     def _dfs(self, i, j):
#         if self._check_no_valid(i, j) or self.grid[i][j] != 1:
#             return 0

#         self.grid[i][j] = -1  # -1表示遍历过这个点了
#         return 1 + self._dfs(i - 1, j) + self._dfs(i, j + 1) \
#             + self._dfs(i + 1, j) + self._dfs(i, j - 1)

#     def _check_no_valid(self, i, j):
#         return i < 0 or i >= self.m or j < 0 or j >= self.n

# class Solution:
#     def dfs(self, grid, cur_i, cur_j):
#         if cur_i < 0 or cur_j < 0 or cur_i == len(grid) or cur_j == len(grid[0]) or grid[cur_i][cur_j] != 1:
#             return 0
#         grid[cur_i][cur_j] = 0
#         ans = 1
#         for di, dj in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
#             next_i, next_j = cur_i + di, cur_j + dj
#             ans += self.dfs(grid, next_i, next_j)
#         return ans

#     def maxAreaOfIsland(self, grid):
#         ans = 0
#         for i, l in enumerate(grid):
#             for j, n in enumerate(l):
#                 ans = max(self.dfs(grid, i, j), ans)
#         return ans

# class Solution:
#     def maxAreaOfIsland(self, grid):
#         ans = 0
#         for i, l in enumerate(grid):
#             for j, n in enumerate(l):
#                 cur = 0
#                 stack = [(i, j)]
#                 while stack:
#                     cur_i, cur_j = stack.pop()
#                     if cur_i < 0 or cur_j < 0 or cur_i == len(grid) or cur_j == len(grid[0]) or grid[cur_i][cur_j] != 1:
#                         continue
#                     cur += 1
#                     grid[cur_i][cur_j] = 0
#                     for di, dj in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
#                         next_i, next_j = cur_i + di, cur_j + dj
#                         stack.append((next_i, next_j))
#                 ans = max(ans, cur)
#         return ans

# import collections

# class Solution:
#     def maxAreaOfIsland(self, grid):
#         ans = 0
#         for i, l in enumerate(grid):
#             for j, n in enumerate(l):
#                 cur = 0
#                 q = collections.deque([(i, j)])
#                 while q:
#                     cur_i, cur_j = q.popleft()
#                     if cur_i < 0 or cur_j < 0 or cur_i == len(grid) or cur_j == len(grid[0]) or grid[cur_i][cur_j] != 1:
#                         continue
#                     cur += 1
#                     grid[cur_i][cur_j] = 0
#                     for di, dj in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
#                         next_i, next_j = cur_i + di, cur_j + dj
#                         q.append((next_i, next_j))
#                 ans = max(ans, cur)
#         return ans

# a = Solution()
# b = a.maxAreaOfIsland([[1,1,0,0,0],[1,1,0,0,0],[0,0,0,1,1],[0,0,0,1,1]])
# print(b)


# class Solution:
#     def merge(self, intervals):
#         res = []
#         intervals.sort()
#         for i in intervals:
#             if not res or res[-1][1] < i[0]:
#                 res.append(i)
#             else:
#                 res[-1][1] = max(i[1],res[-1][1])
#         return res

# class Solution:
#     def insert(self, intervals, newInterval):
#         # init data
#         new_start, new_end = newInterval
#         idx, n = 0, len(intervals)
#         output = []
        
#         # add all intervals starting before newInterval
#         while idx < n and new_start > intervals[idx][0]:
#             output.append(intervals[idx])
#             idx += 1
            
#         # add newInterval
#         # if there is no overlap, just add the interval
#         if not output or output[-1][1] < new_start:
#             output.append(newInterval)
#         # if there is an overlap, merge with the last interval
#         else:
#             output[-1][1] = max(output[-1][1], new_end)
        
#         # add next intervals, merge with newInterval if needed
#         while idx < n:
#             interval = intervals[idx]
#             start, end = interval
#             idx += 1
#             # if there is no overlap, just add an interval
#             if output[-1][1] < start:
#                 output.append(interval)
#             # if there is an overlap, merge with the last interval
#             else:
#                 output[-1][1] = max(output[-1][1], end)
#         return output

# a = Solution()
# b = a.insert([[1,3],[2,6],[8,10],[15,18]], [5,8])
# print(b)

# class Solution:
#     def movingCount(self, m, n, k):
#         def dfs(i, j, si, sj):
#             if not 0 <= i < m or not 0 <= j < n or k < si + sj or (i, j) in visited: return 0
#             visited.add((i,j))
#             return 1 + dfs(i + 1, j, si + 1 if (i + 1) % 10 else si - 8, sj) + dfs(i, j + 1, si, sj + 1 if (j + 1) % 10 else sj - 8)

#         visited = set()
#         return dfs(0, 0, 0, 0)

# class Solution:
#     def movingCount(self, m, n, k):
#         def sumofDigit(x, y):
#             result = 0
#             while x > 0:
#                 result += x % 10
#                 x //= 10
#             while y > 0:
#                 result += y % 10
#                 y //= 10
#             return result
        
#         def dfs(i, j):
#             if i == m or j == n or sumofDigit(i, j) > k or (i, j) in marked:
#                 return 
#             marked.add((i, j))
#             dfs(i + 1, j)
#             dfs(i, j + 1)
            
#         marked = set()
#         dfs(0, 0)
#         return len(marked)

# import queue
# class Solution:
#     def movingCount(self, m, n, k):
#         queue, visited = [(0, 0, 0, 0)], set()
#         while queue:
#             i, j, si, sj = queue.pop()
#             if not 0 <= i < m or not 0 <= j < n or k < si + sj or (i, j) in visited: continue
#             visited.add((i, j))
#             queue.append((i + 1, j, si + 1 if (i + 1) % 10 else si - 8, sj))
#             queue.append((i, j + 1, si, sj + 1 if (j + 1) % 10 else sj - 8))
#         return len(visited)

# import collections

# class Solution:
#     def sum_rc(self,row,col):
#         tmp = 0
#         while row > 0:
#             tmp += row % 10
#             row //= 10
#         while col > 0:
#             tmp += col % 10
#             col //= 10
#         return tmp

#     def movingCount(self, m, n, k):
#         marked = set()  # 将访问过的点添加到集合marked中,从(0,0)开始
#         queue = collections.deque()
#         queue.append((0,0))
#         while queue:
#             x, y = queue.popleft()
#             if (x,y) not in marked and self.sum_rc(x,y) <= k:
#                 marked.add((x,y)) 
#                 for dx, dy in [(1,0),(0,1)]:  # 仅考虑向右和向下即可
#                     if 0 <= x + dx < m and 0 <= y + dy < n:
#                         queue.append((x+dx,y+dy)) 
#         return len(marked)

# a = Solution()
# b = a.movingCount(3,2,3)
# print(b)

# class Solution:
#     def uniquePaths(self, m, n):
#         dp = [[1]*n] + [[1]+[0] * (n-1) for _ in range(m-1)]
#         # print(dp)
#         for i in range(1, m):
#             for j in range(1, n):
#                 dp[i][j] = dp[i-1][j] + dp[i][j-1]
#         return dp[-1][-1]

# class Solution:
#     def uniquePaths(self, m, n):
#         pre = [1] * n
#         cur = [1] * n
#         for i in range(1, m):
#             for j in range(1, n):
#                 cur[j] = pre[j] + cur[j-1]
#             pre = cur[:]
#         return pre[-1]

# class Solution:
#     def uniquePaths(self, m: int, n: int) -> int:
#         cur = [1] * n
#         for i in range(1, m):
#             for j in range(1, n):
#                 cur[j] += cur[j-1]
#         return cur[-1]

# a = Solution()
# b = a.uniquePaths(3,6)
# print(b)

# class Solution(object):
#     def uniquePathsWithObstacles(self, obstacleGrid):
#         """
#         :type obstacleGrid: List[List[int]]
#         :rtype: int
#         """

#         m = len(obstacleGrid)
#         n = len(obstacleGrid[0])

#         # If the starting cell has an obstacle, then simply return as there would be
#         # no paths to the destination.
#         if obstacleGrid[0][0] == 1:
#             return 0

#         # Number of ways of reaching the starting cell = 1.
#         obstacleGrid[0][0] = 1

#         # Filling the values for the first column
#         for i in range(1,m):
#             obstacleGrid[i][0] = int(obstacleGrid[i][0] == 0 and obstacleGrid[i-1][0] == 1)

#         # Filling the values for the first row        
#         for j in range(1, n):
#             obstacleGrid[0][j] = int(obstacleGrid[0][j] == 0 and obstacleGrid[0][j-1] == 1)

#         # Starting from cell(1,1) fill up the values
#         # No. of ways of reaching cell[i][j] = cell[i - 1][j] + cell[i][j - 1]
#         # i.e. From above and left.
#         for i in range(1,m):
#             for j in range(1,n):
#                 if obstacleGrid[i][j] == 0:
#                     obstacleGrid[i][j] = obstacleGrid[i-1][j] + obstacleGrid[i][j-1]
#                 else:
#                     obstacleGrid[i][j] = 0

#         # Return value stored in rightmost bottommost cell. That is the destination.            
#         return obstacleGrid[m-1][n-1]

# a = Solution()
# b = a.uniquePathsWithObstacles([[0,0,1,1],[0,0,0,0],[1,1,1,0],[0,0,0,0]])
# print(b)

# import math
# class Solution:
#     def getPermutation(self, n, k):
#         l=list(range(1,n+1))
#         res=[]
#         while l:
#             a=math.factorial(len(l)-1)
#             tmp=math.ceil(k/a)-1
#             value=l[tmp]
#             res.append(value)
#             l.remove(value)
#             k=k-tmp*a
#         res=list(map(str,res+l))
#         return ''.join(res)

# a = Solution()
# b = a.getPermutation(5, 46)
# print(b)

# import heapq

# class Solution:
#     def getLeastNumbers(self, arr, k):
#         if k == 0:
#             return list()

#         hp = [-x for x in arr[:k]]
#         heapq.heapify(hp)
#         for i in range(k, len(arr)):
#             if -hp[0] > arr[i]:
#                 heapq.heappop(hp)
#                 heapq.heappush(hp, -arr[i])
#         ans = [-x for x in hp]
#         return ans

# import random

# class Solution:
#     def partition(self, nums, l, r):
#         pivot = nums[r]
#         i = l - 1
#         for j in range(l, r):
#             if nums[j] <= pivot:
#                 i += 1
#                 nums[i], nums[j] = nums[j], nums[i]
#         nums[i + 1], nums[r] = nums[r], nums[i + 1]
#         return i + 1

#     def randomized_partition(self, nums, l, r):
#         i = random.randint(l, r)
#         nums[r], nums[i] = nums[i], nums[r]
#         return self.partition(nums, l, r)

#     def randomized_selected(self, arr, l, r, k):
#         pos = self.randomized_partition(arr, l, r)
#         num = pos - l + 1
#         if k < num:
#             self.randomized_selected(arr, l, pos - 1, k)
#         elif k > num:
#             self.randomized_selected(arr, pos + 1, r, k - num)

#     def getLeastNumbers(self, arr, k):
#         if k == 0:
#             return list()
#         self.randomized_selected(arr, 0, len(arr) - 1, k)
#         return arr[:k]

# class Solution:
#     def getLeastNumbers(self, arr, k):
#         def qsort(l,r):
#             i=l-1
#             for j in range(l,r):
#                 if arr[j]<=arr[r]:
#                     i+=1
#                     arr[i],arr[j]=arr[j],arr[i]
#             i+=1
#             arr[i],arr[r]=arr[r],arr[i]
#             return i

#         def helper(l,r,k):
#             p=qsort(l,r)
            
#             if k<p-l+1:
#                 helper(l,p-1,k)
#             elif k>p-l+1:
#                 helper(p+1,r,k-(p-l+1))

#         if k==0:
#             return []
#         helper(0,len(arr)-1,k)
#         return arr[:k]

# a = Solution()
# b = a.getLeastNumbers([1,5,4,6,2,8,7,3,9],3)
# print(b)

# class Solution:
#     def translateNum(self, num):
#         self.s = set()
#         num = str(num)
#         n = len(num)

#         def dfs(cur, tmp):
#             if cur==n:
#                 self.s.add(tuple(tmp))
#                 return
#             for i in range(cur, n):
#                 if len(num[cur:i+1])>1 and num[cur]=='0':
#                     continue
#                 if 0<=int(num[cur:i+1])<=25:
#                     dfs(i+1, tmp+[num[cur:i+1]])
        
#         dfs(0, [])
#         return len(self.s)

# class Solution:
#     def translateNum(self, num: int) -> int:
#         str_num = str(num)
#         n = len(str_num)
#         dp = [1 for _ in range(n + 1)] 
#         for i in range(2, n + 1):
#             if str_num[i - 2] == '1' or \
#             (str_num[i - 2] == '2' and str_num[i - 1] < '6'):
#                 dp[i] = dp[i - 2] + dp[i - 1]
#             else:
#                 dp[i] = dp[i - 1]
#         return dp[n]

# class Solution:
#     def translateNum(self, num: int) -> int:
#         def backtrack(s, idx):
#             n = len(s)
#             if idx == n: return 1
#             if idx == n - 1 or s[idx] == '0' or s[idx:idx + 2] > '25':
#                 return backtrack(s, idx + 1)
#             else:
#                 return backtrack(s, idx + 1) + backtrack(s, idx + 2)
        
#         s = str(num)
#         return backtrack(s, 0)

# a = Solution()
# b = a.translateNum(12258)
# print(b)

# class Solution:
#     def minPathSum(self, grid):
#         ans = [[0] * len(grid[0]) for i in range(len(grid))]
#         ans[0][0]= grid[0][0]
#         for i in range(1,len(grid)):
#             print(grid[0][i])
#             ans[0][i] = grid[0][i] + grid[0][i-1]
#         # for j in range(1, len(grid[0])):
#         #     ans[j][0] = grid[j][0] + grid[j-1][0]
#         print(ans)

# class Solution:
#     def minPathSum(self, grid):
#         for i in range(1,len(grid[0])):
#             grid[0][i] = grid[0][i] + grid[0][i-1]
#         for j in range(1, len(grid)):
#             grid[j][0] = grid[j][0] + grid[j-1][0]
#         for i in range(1, len(grid)):
#             for j in range(1, len(grid[0])):
#                 grid[i][j] = grid[i][j] + min(grid[i-1][j], grid[i][j-1])
#         return grid[-1][-1]

# a = Solution()
# b = a.minPathSum([[1,2,5],[3,2,1]])
# print(b)

# class Solution:
#     def minIncrementForUnique(self, A):
#         A.sort()
#         count = 0
#         for i in range(1, len(A)):
#             while A[i] <= A[i-1]:
#                 count += A[i-1] - A[i] + 1
#                 A[i] += A[i-1] - A[i] + 1
#         return count

# a = Solution()
# b = a.minIncrementForUnique([3,2,1,2,1,7])
# print(b)

# word1 = 'abcde'
# word2 = 'abcdef'
# n1, n2 = len(word1), len(word2) 
# dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
# # 第一行
# for j in range(1, n2 + 1):
#     dp[0][j] = j
# (3236)# 第一列
# for i in range(1, n1 + 1):
#     dp[i][0] = i
    
# for i in range(1, n1 + 1):
#     for j in range(1, n2 + 1):
#         if word1[i-1] == word2[j-1]:
#             dp[i][j] = dp[i-1][j-1]
#         else:
#             dp[i][j] = min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1] ) + 1     
# print(dp[-1][-1])

# class Solution:
#     def permutation(self, s):
#         if len(s) == 1: return [s]
#         res = []
#         for i, x in enumerate(s):
#             n = s[:i] + s[i+1:]
#             for y in self.permutation(n):
#                 res.append(x+y)
#         return list(set(res))

# a = Solution()
# b = a.permutation('abc')
# print(b)

# class Solution:
#     def search(self, nums, target):
#         i, j = 0, len(nums) - 1
#         while i <= j:
#             m = (i + j) // 2
#             if nums[m] <= target: i = m + 1
#             else: j = m - 1
#         right = i

#         i, j = 0, len(nums) - 1
#         while i <= j:
#             m = (i + j) // 2
#             if nums[m] < target: i = m + 1
#             else: j = m - 1
#         left = j
        
#         return right - left - 1

# a = Solution()
# b = a.search([5,6,7,8,8,8,9,9,9,10,10,10], 8)
# print(b)

# class Solution(object):
#     def simplifyPath(self, path):
#         stack = []
#         for i in path.split('/'):
#             if i not in ['', '.', '..']:
#                 stack.append(i)
#             elif i == '..' and stack:
#                 stack.pop()
#         return "/" + "/".join(stack)

# a = Solution()
# b = a.simplifyPath('/a/b/c//.//')
# print(b)

# import collections
# from functools import reduce

# class Solution(object):
#     def hasGroupsSizeX(self, deck):
#         from fractions import gcd
#         vals = collections.Counter(deck).values()
#         return reduce(gcd, vals) >= 2

# a = Solution()
# b = a.hasGroupsSizeX([1,2,3,4,4,3,2,1])
# print(b)

# import collections

# class Solution(object):
#     def hasGroupsSizeX(self, deck):
#         count = collections.Counter(deck)
#         N = len(deck)
#         for X in range(2, N+1):
#             if N % X == 0:
#                 if all(v % X == 0 for v in count.values()):
#                     return True
#         return False

# a = Solution()
# b = a.hasGroupsSizeX([1,2,3,4,4,3,2,1])
# print(b)

# class Solution:
#     def minimumLengthEncoding(self, words):
#         good = set(words)
#         for word in words:
#             for k in range(1, len(word)):
#                 good.discard(word[k:])

#         return sum(len(word) + 1 for word in good)

# a = Solution()
# b = a.minimumLengthEncoding(["time", "me", "bell"])
# print(b)

# import collections

# class Solution:
#     def maxDistance(self, grid):
#         rows = len(grid)
#         cols = len(grid[0])
#         start = []
#         for i in range(rows): # 将所有起点存入 start 数组
#             for j in range(cols):
#                 if grid[i][j] == 1:
#                     start.append((i, j, 0))
        
#         if len(start) == 0 or len(start) == rows * cols: # 特判
#             return -1

#         queue = collections.deque(start) # 队列初始化
#         dr = [0, 1, 0, -1] # 建立方向数组
#         dc = [1, 0, -1, 0]
#         while queue:
#             i, j, dis = queue.popleft()
#             for d in range(4): # 四个方向
#                 x = i + dr[d]
#                 y = j + dc[d]
#                 if x < 0 or y < 0 or x == rows or y == cols or grid[x][y] == 1: 
#                     continue
#                 queue.append((x, y, dis + 1))
#                 grid[x][y] = 1 # 访问过的位置标记为 1
                
#         return dis

# class Solution:
#     def maxDistance(self, grid):
#         n, m = len(grid), len(grid[0])
#         dis = [[float("inf") for j in range(m+2)] for i in range(n+2)]
#         for i in range(1, n+1):
#             for j in range(1, m+1):
#                 if grid[i-1][j-1]:
#                     dis[i][j] = 0
#                 else:
#                     dis[i][j] = min(dis[i-1][j], dis[i][j-1]) + 1
#         res = -1
#         for i in range(n, 0, -1):
#             for j in range(m, 0, -1):
#                 if grid[i-1][j-1]:
#                     dis[i][j] = 0
#                 else:
#                     dis[i][j] = min(dis[i+1][j]+1, dis[i][j+1]+1, dis[i][j])
#                     res = max(dis[i][j], res)
#         return res if res != -1 and res != float("inf") else -1

# a = Solution()
# b = a.maxDistance([[1,0,1],[0,0,0],[1,0,1]])
# print(b)

# class Solution:
#     def hammingWeight(self, n):
#         summ = 0
#         while n != 0:
#             summ += 1
#             n &= n - 1
#         return summ

# class Solution:
#     def hammingWeight(self, n):
#         ans = 0
#         while n:
#             if n & 1: ans += 1
#             n >>= 1
#         return ans

# a = Solution()
# b = a.hammingWeight(1000000101010)
# print(b)

# import random
# class Solution:
#     def randomized_partition(self, nums, l, r):
#         pivot = random.randint(l, r)
#         nums[pivot], nums[r] = nums[r], nums[pivot]
#         i = l - 1
#         for j in range(l, r):
#             if nums[j] < nums[r]:
#                 i += 1
#                 nums[j],nums[i] = nums[i], nums[j]
#         i += 1
#         nums[i], nums[r] = nums[r], nums[i]
#         return i
            
#     def randomized_quicksort(self, nums, l, r):
#         if r - l <= 0:
#             return
#         mid = self.randomized_partition(nums, l, r)
#         self.randomized_quicksort(nums, l, mid - 1)
#         self.randomized_quicksort(nums, mid + 1, r)
#     def sortArray(self, nums):
#         self.randomized_quicksort(nums, 0, len(nums) - 1)
#         return nums

# class Solution:
#     def max_heapify(self, heap, root, heap_len):
#         p = root
#         while p * 2 + 1 < heap_len:
#             l, r = p * 2 + 1, p * 2 + 2
#             if heap_len <= r or heap[r] < heap[l]:
#                 nex = l
#             else:
#                 nex = r
#             if heap[p] < heap[nex]:
#                 heap[p], heap[nex] = heap[nex], heap[p]
#                 p = nex
#             else:
#                 break

#     def build_heap(self, heap):
#         for i in range(len(heap) - 1, -1, -1):
#             self.max_heapify(heap, i, len(heap))

#     def heap_sort(self, nums):
#         self.build_heap(nums)
#         for i in range(len(nums) - 1, -1, -1):
#             nums[i], nums[0] = nums[0], nums[i]
#             self.max_heapify(nums, 0, i)

#     def sortArray(self, nums):
#         self.heap_sort(nums)
#         return nums

class Solution:
    def merge_sort(self, nums, l, r):
        if l == r:
            return
        mid = (l + r) // 2
        self.merge_sort(nums, l, mid)
        self.merge_sort(nums, mid + 1, r)
        tmp = []
        i, j = l, mid + 1
        while i <= mid or j <= r:
            if i > mid or (j <= r and nums[j] < nums[i]):
                tmp.append(nums[j])
                j += 1
            else:
                tmp.append(nums[i])
                i += 1
        nums[l: r + 1] = tmp

    def sortArray(self, nums):
        self.merge_sort(nums, 0, len(nums) - 1)
        return nums

# class Solution:
#     def sortArray(self, nums):
#         for i in range(len(nums)):
#             for j in range(len(nums)-1, i, -1):
#                 if nums[j] < nums[j-1]:
#                     nums[j], nums[j-1] = nums[j-1], nums[j]
#         return nums

a = Solution()
b = a.sortArray([1,2,6,5,7,8,4,9,3])
print(b)

# def insertion_sort(nums):
#     n = len(nums)
#     for i in range(1, n):
#         while i > 0 and nums[i - 1] > nums[i]:
#             nums[i - 1], nums[i] = nums[i], nums[i - 1]
#             i -= 1
#     return nums

# b = insertion_sort([5,2,3,1,6])
# print(b)

# def selection_sort(nums):
#     n = len(nums)
#     for i in range(n-1):
#         for j in range(i+1, n):
#             if nums[i] > nums[j]:
#                 nums[i], nums[j] = nums[j], nums[i]
#     return nums

# b = selection_sort([1,2,6,5,7,8,4,9,3])
# print(b)