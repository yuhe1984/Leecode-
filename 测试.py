# class Solution2:
#     def twoSum(nums, target):
#         hashmap={}
#         for ind,num in enumerate(nums):
#             hashmap[num] = ind
#         for i,num in enumerate(nums):
#             j = hashmap.get(target - num)
#             if j is not None and i!=j:
#                 return [i,j]


# b = Solution2
# nums = [3,2,4]
# target = 6
# c = b.twoSum(nums,target)
# print(c)


# # Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# class Solution:
#     def addTwoNumbers(self, l1, l2):
#         prenode = ListNode(0)
#         lastnode = prenode
#         val = 0
#         while val or l1 or l2:
#             val, cur = divmod(val + (l1.val if l1 else 0) + (l2.val if l2 else 0), 10)
#             lastnode.next = ListNode(cur)
#             lastnode = lastnode.next
#             l1 = l1.next if l1 else None
#             l2 = l2.next if l2 else None
#         return prenode.next


# def generateList(l: list) -> ListNode:
#     prenode = ListNode(0)
#     lastnode = prenode
#     for val in l:
#         lastnode.next = ListNode(val)
#         lastnode = lastnode.next
#     return prenode.next

# def printList(l: ListNode):
#     while l:
#         print("%d, " %(l.val), end = '')
#         l = l.next
#     print('')

# if __name__ == "__main__":
#     l1 = generateList([1, 5, 8])
#     l2 = generateList([9, 1, 2, 9])
#     printList(l1)
#     printList(l2)
#     s = Solution()
#     sum = s.addTwoNumbers(l1, l2)
#     printList(sum)


# class Solution:
#     def lengthOfLongestSubstring(s: str) -> int:
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

# a = Solution
# s = "pwwkew"
# b = a.lengthOfLongestSubstring(s)
# print(b)


# money=[1,5,10,20,50,100]
# n = 10
# li=[]
# for i in range(n+1):
#     li.append(0)
# li[0]=1
# for i in money:
#     for j in range(n+1):
#         if j>=i:
#             li[j]=li[j]+li[j-i]
#     print(li)


# def FindGreatestSumOfSubArray(array):
#         f = array[0]
#         res = array[0]
#         for i in range(1,len(array)):
#             f = max(f+array[i],array[i])
#             res = max(res,f)
#         return res

# array = [6,-3,-2,7,-15,1,2,2]
# a = FindGreatestSumOfSubArray(array)
# print(a)


# class Solution:
#     def convert(self, s: str, numRows: int) -> str:
#         if numRows < 2: return s
#         res = ["" for _ in range(numRows)]
#         i, flag = 0, -1
#         for c in s:
#             res[i] += c
#             if i == 0 or i == numRows - 1: flag = -flag
#             i += flag
#         print(res)
#         return "".join(res)

# s = "LEFTCOD"
# numRows = 3
# a = Solution
# a.convert([],s,numRows)


# class Solution:
#     def romanToInt(self, s: str) -> int:
#         d = {'I':1, 'IV':3, 'V':5, 'IX':8, 'X':10, 'XL':30, 'L':50, 'XC':80, 'C':100, 'CD':300, 'D':500, 'CM':800, 'M':1000}
#         return sum(d.get(s[max(i-1, 0):i+1], d[n]) for i, n in enumerate(s))

# s = 'LVIII'
# a = Solution
# b = a.romanToInt([],s)
# print(b)


# class Solution:
#     def threeSum(self, nums):
        
#         n=len(nums)
#         res=[]
#         if(not nums or n<3):
#             return []
#         nums.sort()
#         res=[]
#         for i in range(n):
#             if(nums[i]>0):
#                 return res
#             if(i>0 and nums[i]==nums[i-1]):
#                 continue
#             L=i+1
#             R=n-1
#             while(L<R):
#                 if(nums[i]+nums[L]+nums[R]==0):
#                     res.append([nums[i],nums[L],nums[R]])
#                     while(L<R and nums[L]==nums[L+1]):
#                         L=L+1
#                     while(L<R and nums[R]==nums[R-1]):
#                         R=R-1
#                     L=L+1
#                     R=R-1
#                 elif(nums[i]+nums[L]+nums[R]>0):
#                     R=R-1
#                 else:
#                     L=L+1
#         return res


# a = Solution
# b = a.threeSum([],[1,2,-3,0,-1,-2,2,1,5])
# print(b)


# class Solution:
#     def letterCombinations(self, digits):
#         """
#         :type digits: str
#         :rtype: List[str]
#         """
#         phone = {'2': ['a', 'b', 'c'],
#                  '3': ['d', 'e', 'f'],
#                  '4': ['g', 'h', 'i'],
#                  '5': ['j', 'k', 'l'],
#                  '6': ['m', 'n', 'o'],
#                  '7': ['p', 'q', 'r', 's'],
#                  '8': ['t', 'u', 'v'],
#                  '9': ['w', 'x', 'y', 'z']}
                
#         def backtrack(combination, next_digits):
#             # if there is no more digits to check
#             if len(next_digits) == 0:
#                 # the combination is done
#                 output.append(combination)
#             # if there are still digits to check
#             else:
#                 # iterate over all letters which map 
#                 # the next available digit
#                 for letter in phone[next_digits[0]]:
#                     # append the current letter to the combination
#                     # and proceed to the next digits
#                     backtrack(combination + letter, next_digits[1:])
                    
#         output = []
#         if digits:
#             backtrack("", digits)
#         return output

# a = Solution
# b = a.letterCombinations([],'235')
# print(b)


# class Solution:
#     def removeDuplicates(self, nums):
#         if (len(nums) == 0): return 0
#         i = 0
#         for j in range(len(nums)):
#             if (nums[j] != nums[i]):
#                 i += 1
#                 nums[i] = nums[j]
#         return i + 1


# a = Solution
# b = a.removeDuplicates([],[0,0,1,1,2,2,3,3,4,4,5,5])
# print(b)

# class Solution:
#     def divide(self, dividend, divisor):
#         sign = (dividend > 0) ^ (divisor > 0)
#         dividend = abs(dividend)
#         divisor = abs(divisor)
#         count = 0
#         #把除数不断左移，直到它大于被除数
#         while dividend >= divisor:
#             count += 1
#             divisor <<= 1
#         result = 0
#         while count > 0:
#             count -= 1
#             divisor >>= 1
#             if divisor <= dividend:
#                 result += 1 << count #这里的移位运算是把二进制（第count+1位上的1）转换为十进制
#                 dividend -= divisor
#         if sign: result = -result
#         return result if -(1<<31) <= result <= (1<<31)-1 else (1<<31)-1 

# a = Solution
# b = a.divide([],45,2)
# print(b)

# def perm(n,begin,end):
#     ans = []
#     if begin>=end:
#         ans.append(''.join((x for x in n)))
#     else:
#         i=begin
#         for num in range(begin,end):
#             n[num],n[i]=n[i],n[num]
#             perm(n,begin+1,end)
#             n[num],n[i]=n[i],n[num]
#     return ans

# n = ['a','b','c','d']
# b = perm(n,0,4)
# print(b)

# class Solution(object):
#     def search(self, nums, target):
#         """
#         :type nums: List[int]
#         :type target: int
#         :rtype: int
#         """
#         def search_erfen(nums,left,right,target):
#             if(left > right):
#                 return -1
#             elif(left == right):
#                 if(nums[left] == target):
#                     return left
#                 else:
#                     return -1
#             elif(left < right):
#                 if (target < nums[left] or nums[right] < target):
#                     return -1
#                 while(left < right):
#                     mid = int((left + right)/2)
#                     if(nums[mid] < target):
#                         left = mid
#                         if(right == left + 1):
#                             if(nums[right] == target):
#                                 return right
#                             else:
#                                 return -1
#                     elif(nums[mid] > target):
#                         right = mid
#                     elif(nums[mid] == target):
#                         return mid

#         left = 0
#         right = len(nums)-1
#         if(left > right):
#             return -1
#         elif(left == right):
#             if(nums[left] == target):
#                 return left
#             else:
#                 return -1
#         elif(left < right):
#             while(left < right):
#                 if(right == left+1):
#                     if(nums[left] == target):
#                         return left
#                     elif(nums[right] == target):
#                         return right
#                     else:
#                         return -1

#                 mid = int((left + right)/2)
#                 if(nums[mid] < nums[right]):
#                     ans = search_erfen(nums,mid,right,target)
#                     if (ans != -1):
#                         return ans
#                     right = mid
#                 elif(nums[left] < nums[mid]):                   
#                     ans = search_erfen(nums,left,mid,target)
#                     if (ans != -1):
#                         return ans
#                     left = mid


# a = Solution()
# b = a.search([4,5,6,7,8,1,2,3],2)
# print(b)

# class Solution:
#     def nextPermutation(self, nums):
#         """
#         Do not return anything, modify nums in-place instead.
#         """
#         if not nums:
#             return None
#         n = len(nums)
#         if n<2:
#             return nums
#         def reverse(ind):  
#             """
#             把nums[ind:]进行翻转
#             """
#             a, b = ind, n-1
#             while a<b:
#                 nums[a], nums[b] = nums[b], nums[a]
#                 a += 1
#                 b -= 1

#         for i in range(n-1, 0, -1):
#             if nums[i-1]<nums[i]:
#                 if i==n-1:  
#                     # 如果刚好是最后两位则将其换位后输出
#                     nums[i-1], nums[i] = nums[i], nums[i-1]
#                     return nums
#                 else:
#                     if nums[i-1]<nums[n-1]:
#                         # 如果nums[i-1]小于nums[i:]中最小值，则将原最小值换过来，再翻转
#                         nums[i-1], nums[n-1] = nums[n-1], nums[i-1]
#                         reverse(i) # i begin
#                         return nums
#                     else:
#                         for j in range(i, n):
#                             # 如果nums[i-1]后面有更小的，则交换再翻转
#                             if nums[i-1]>=nums[j]:
#                                 nums[i-1], nums[j-1] = nums[j-1], nums[i-1]
#                                 reverse(i)
#                                 return nums
#         reverse(0)  # 如果是最大的，则翻转成最小的               
#         return nums

# a = Solution
# b = a.nextPermutation([],[1,4,9,8,7,3])
# print(b)

# class Solution:
#     # returns leftmost (or rightmost) index at which `target` should be inserted in sorted
#     # array `nums` via binary search.
#     def extreme_insertion_index(self, nums, target, left):
#         lo = 0
#         hi = len(nums)

#         while lo < hi:
#             mid = (lo + hi) // 2
#             if nums[mid] > target or (left and target == nums[mid]):
#                 hi = mid
#             else:
#                 lo = mid+1

#         return lo


#     def searchRange(self, nums, target):
#         left_idx = Solution.extreme_insertion_index([],nums, target, True)

#         # assert that `left_idx` is within the array bounds and that `target`
#         # is actually in `nums`.
#         if left_idx == len(nums) or nums[left_idx] != target:
#             return [-1, -1]

#         return [left_idx, Solution.extreme_insertion_index([],nums, target, False)-1]


# a = Solution
# b = a.searchRange([],[5,6,6,7,7,10],6)
# print(b)


# class Solution:
#     def isValidSudoku(self, board):
#         """
#         :type board: List[List[str]]
#         :rtype: bool
#         """
#         # init data
#         rows = [{} for i in range(9)]
#         columns = [{} for i in range(9)]
#         boxes = [{} for i in range(9)]

#         # validate a board
#         for i in range(9):
#             for j in range(9):
#                 num = board[i][j]
#                 if num != '.':
#                     num = int(num)
#                     box_index = (i // 3 ) * 3 + j // 3
                    
#                     # keep the current cell value
#                     rows[i][num] = rows[i].get(num, 0) + 1
#                     columns[j][num] = columns[j].get(num, 0) + 1
#                     boxes[box_index][num] = boxes[box_index].get(num, 0) + 1
                    
#                     # check if this value has been already seen before
#                     if rows[i][num] > 1 or columns[j][num] > 1 or boxes[box_index][num] > 1:
#                         return False
#         return True


# a = Solution
# board = [
#     ["8","3",".",".","7",".",".",".","."],
#     ["6",".",".","1","9","5",".",".","."],
#     [".","9","8",".",".",".",".","6","."],
#     ["8",".",".",".","6",".",".",".","3"],
#     ["4",".",".","8",".","3",".",".","1"],
#     ["7",".",".",".","2",".",".",".","6"],
#     [".","6",".",".",".",".","2","8","."],
#     [".",".",".","4","1","9",".",".","5"],
#     [".",".",".",".","8",".",".","7","9"]
# ]
# b = a.isValidSudoku([],board)
# print(b)

# class Solution:
#     def majorityElement(self, nums, lo=0, hi=None):
#         def majority_element_rec(lo, hi):
#             # base case; the only element in an array of size 1 is the majority
#             # element.
#             if lo == hi:
#                 return nums[lo]

#             # recurse on left and right halves of this slice.
#             mid = (hi-lo)//2 + lo
#             left = majority_element_rec(lo, mid)
#             right = majority_element_rec(mid+1, hi)

#             # if the two halves agree on the majority element, return it.
#             if left == right:
#                 return left

#             # otherwise, count each element and return the "winner".
#             left_count = sum(1 for i in range(lo, hi+1) if nums[i] == left)
#             right_count = sum(1 for i in range(lo, hi+1) if nums[i] == right)

#             return left if left_count > right_count else right

#         return majority_element_rec(0, len(nums)-1)

# a = Solution
# b = a.majorityElement([],[1,2,2,3,3,3,4,4,4,4,4,5,5,5,5,5,6])
# print(b)

# class Solution:
#     def solveSudoku(self, board):
#         row = [set(range(1, 10)) for _ in range(9)]  # 行剩余可用数字
#         col = [set(range(1, 10)) for _ in range(9)]  # 列剩余可用数字
#         block = [set(range(1, 10)) for _ in range(9)]  # 块剩余可用数字

#         empty = []  # 收集需填数位置
#         for i in range(9):
#             for j in range(9):
#                 if board[i][j] != '.':  # 更新可用数字
#                     val = int(board[i][j])
#                     row[i].remove(val)
#                     col[j].remove(val)
#                     block[(i // 3)*3 + j // 3].remove(val)
#                 else:
#                     empty.append((i, j))

#         def backtrack(iter=0):
#             if iter == len(empty):  # 处理完empty代表找到了答案
#                 return True
#             i, j = empty[iter]
#             b = (i // 3)*3 + j // 3
#             for val in row[i] & col[j] & block[b]:
#                 row[i].remove(val)
#                 col[j].remove(val)
#                 block[b].remove(val)
#                 board[i][j] = str(val)
#                 if backtrack(iter+1):
#                     return True
#                 row[i].add(val)  # 回溯
#                 col[j].add(val)
#                 block[b].add(val)
#             return False
#         backtrack()

# a = Solution
# board = [
#     ["5","3",".",".","7",".",".",".","."],
#     ["6",".",".","1","9","5",".",".","."],
#     [".","9","8",".",".",".",".","6","."],
#     ["8",".",".",".","6",".",".",".","3"],
#     ["4",".",".","8",".","3",".",".","1"],
#     ["7",".",".",".","2",".",".",".","6"],
#     [".","6",".",".",".",".","2","8","."],
#     [".",".",".","4","1","9",".",".","5"],
#     [".",".",".",".","8",".",".","7","9"]
# ]
# b = a.solveSudoku([],board)
# print(b)

# class Solution:
#     def countAndSay(self, n: int) -> str:
#         if n == 1:
#             return '1'
#         res = '1'
#         for i in range(2, n+1):
#             ans = ''
#             j = 0
#             while j < len(res):
#                 k = j
#                 while k < len(res)-1 and res[k] == res[k+1]:
#                     k+=1
#                 ans += str(k+1-j)+res[k]
#                 j = k+1
#             res = ans
#         return res

# a = Solution
# b = a.countAndSay([],6)
# print(b)


# from typing import List
# class Solution:
#     def combinationSum(self, candidates, target):
#         size = len(candidates)
#         if size == 0:
#             return []
            
#         # 剪枝的前提是数组元素排序
#         # 深度深的边不能比深度浅的边还小
#         # 要排序的理由：1、前面用过的数后面不能再用；2、下一层边上的数不能小于上一层边上的数。
#         candidates.sort()
#         # 在遍历的过程中记录路径，一般而言它是一个栈
#         path = []
#         res = []
#         # 注意要传入 size ，在 range 中， size 取不到
#         self.__dfs(candidates, 0, size, path, res, target)
#         return res

#     def __dfs(self, candidates, begin, size, path, res, target):
#         # 先写递归终止的情况
#         if target == 0:
#             # Python 中可变对象是引用传递，因此需要将当前 path 里的值拷贝出来
#             # 或者使用 path.copy()
#             res.append(path[:])

#         for index in range(begin, size):
#             residue = target - candidates[index]
#             # “剪枝”操作，不必递归到下一层，并且后面的分支也不必执行
#             if residue < 0:
#                 break
#             path.append(candidates[index])
#             # 因为下一层不能比上一层还小，起始索引还从 index 开始
#             self.__dfs(candidates, index, size, path, res, residue)
#             path.pop()


# if __name__ == '__main__':
#     candidates = [2, 3, 6, 7]
#     target = 7
#     solution = Solution()
#     result = solution.combinationSum(candidates, target)
#     print(result)

# from typing import List

# class Solution:
#     def combinationSum2(self, candidates, target):
#         size = len(candidates)
#         if size == 0:
#             return []

#         # 剪枝的前提是数组元素排序
#         # 深度深的边不能比深度浅的边还小
#         # 要排序的理由：1、前面用过的数后面不能再用；2、下一层边上的数不能小于上一层边上的数。
#         candidates.sort()
#         # 在遍历的过程中记录路径，一般而言它是一个栈
#         path = []
#         res = []
#         # 注意要传入 size ，在 range 中， size 取不到
#         self.__dfs(candidates, 0, size, path, res, target)
                
        
#         return res

#     def __dfs(self, candidates, begin, size, path, res, target):
#         # 先写递归终止的情况
#         if target == 0:
#             # Python 中可变对象是引用传递，因此需要将当前 path 里的值拷贝出来
#             # 或者使用 path.copy()
#             res.append(path[:])

#         for index in range(begin, size):
#             residue = target - candidates[index]
#             #  “剪枝”操作，不必递归到下一层，并且后面的分支也不必执行
#             if residue < 0:
#                 break
#             path.append(candidates[index])
#             # 因为下一层不能比上一层还小，起始索引还从 index 开始
#             self.__dfs(candidates, index+1, size, path, res, residue)
#             path.pop()
# if __name__ =='__main__':
#     candidates = [10,1,2,7,6,1,5]
#     target = 8
#     a = Solution()
#     b = a.combinationSum2(candidates, target)
#     print(b)

# from typing import List


# class Solution:

#     def combinationSum2(self, candidates, target):
#         def dfs(begin, path, residue):
#             if residue == 0:
#                 res.append(path[:])
#                 return

#             for index in range(begin, size):
#                 if candidates[index] > residue:
#                     break

#                 if index > begin and candidates[index - 1] == candidates[index]:
#                     continue

#                 path.append(candidates[index])
#                 dfs(index + 1, path, residue - candidates[index])
#                 path.pop()

#         size = len(candidates)
#         if size == 0:
#             return []

#         candidates.sort()
#         res = []
#         dfs(0, [], target)
#         return res


# a = Solution
# b = a.combinationSum2([],[10,1,2,6,7,2,1,5],8)
# print(b)


# class Solution:
#     def firstMissingPositive(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: int
#         """
#         n = len(nums)
        
#         # 基本情况
#         if 1 not in nums:
#             return 1
        
#         # nums = [1]
#         if n == 1:
#             return 2
        
#         # 用 1 替换负数，0，和大于 n 的数
#         # 在转换以后，nums 只会包含正数
#         for i in range(n):
#             if nums[i] <= 0 or nums[i] > n:
#                 nums[i] = 1
        
#         # 使用索引和数字符号作为检查器
#         # 例如，如果 nums[1] 是负数表示在数组中出现了数字 `1`
#         # 如果 nums[2] 是正数 表示数字 2 没有出现
#         for i in range(n): 
#             a = abs(nums[i])
#             # 如果发现了一个数字 a - 改变第 a 个元素的符号
#             # 注意重复元素只需操作一次
#             if a == n:
#                 nums[0] = - abs(nums[0])
#             else:
#                 nums[a] = - abs(nums[a])
            
#         # 现在第一个正数的下标
#         # 就是第一个缺失的数
#         for i in range(1, n):
#             if nums[i] > 0:
#                 return i
        
#         if nums[0] > 0:
#             return n
            
#         return n + 1

# class Solution:
#     def firstMissingPositive(self, nums):
#         # 保证有1
#         if 1 not in nums:
#             return 1
        
#         n = len(nums)
        
#         # 保证都在1~n的范围内
#         for i in range(n):
#             if nums[i] <= 0 or nums[i] > n:
#                 nums[i] = 1
        
#         # 以自身正负为bitmap，标记
#         for i in range(n):
#             if nums[abs(nums[i])-1] > 0:
#                 nums[abs(nums[i])-1] = -nums[abs(nums[i])-1]
        
#         # 找到第一个为正的索引，即没有出现的最小正数
#         for i in range(n):
#             if nums[i] > 0:
#                 return i+1
        
#         # 全为负
#         return n+1

# a = Solution()
# b = a.firstMissingPositive([0,1,2,3,6,7,8,9])
# print(b)

# class Solution(object):
#     def isMatch(self, text, pattern):
#         memo = {}
#         def dp(i, j):
#             if (i, j) not in memo:
#                 if j == len(pattern):
#                     ans = i == len(text)
#                 else:
#                     first_match = i < len(text) and pattern[j] in {text[i], '.'}
#                     if j+1 < len(pattern) and pattern[j+1] == '*':
#                         ans = dp(i, j+2) or first_match and dp(i+1, j)
#                     else:
#                         ans = first_match and dp(i+1, j+1)

#                 memo[i, j] = ans
#             return memo[i, j]

#         return dp(0, 0)


# a = Solution()
# b = a.isMatch("mississippi","mis*is*p*.")
# print(b)

# class Solution:
#     def myPow(self, x: float, n: int) -> float:
#         if n < 0:
#             x = 1 / x
#             n = -n
#         elif not n:
#             return 1
        
#         ans = 1
#         while n > 1:
#             if n & 1:
#                 ans *= x
#                 n -= 1
#             else:
#                 x = x * x
#                 n //= 2
#         ans *= x
#         return ans

# a = Solution()
# b = a.myPow(2,10)
# print(b)

class Solution:
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def backtrack(first = 0):
            # if all integers are used up
            if first == n:  
                output.append(nums[:])
                return 
            for i in range(first, n):
                # place i-th integer first 
                # in the current permutation
                nums[first], nums[i] = nums[i], nums[first]
                # use next integers to complete the permutations
                backtrack(first + 1)
                # backtrack
                nums[first], nums[i] = nums[i], nums[first]
        
        n = len(nums)
        output = []
        backtrack()
        return output


# class Solution:
#     def permute(self, nums):
#         res = []
#         def backtrack(nums, tmp):
#             if not nums:
#                 res.append(tmp)
#                 return 
#             for i in range(len(nums)):
#                 backtrack(nums[:i] + nums[i+1:], tmp + [nums[i]])
#         backtrack(nums, [])
#         return res


a = Solution()
b = a.permute([1,2,3])
print(b)


# from typing import List


# class Solution:
#     def permute(self, nums):
#         def dfs(nums, size, depth, path, used, res):
#             if depth == size:
#                 res.append(path[:])
#                 return

#             for i in range(size):
#                 if not used[i]:
#                     used[i] = True
#                     path.append(nums[i])

#                     dfs(nums, size, depth + 1, path, used, res)

#                     used[i] = False
#                     path.pop()

#         size = len(nums)
#         if len(nums) == 0:
#             return []

#         used = [False for _ in range(size)]
#         res = []
#         dfs(nums, size, 0, [], used, res)
#         return res


# if __name__ == '__main__':
#     nums = [1, 2, 3]
#     solution = Solution()
#     res = solution.permute(nums)
#     print(res)

# 后有序数组中元素出列的时候，计算逆序个数

# from typing import List


# class Solution:

#     def reversePairs(self, nums):
#         size = len(nums)
#         if size < 2:
#             return 0

#         # 用于归并的辅助数组
#         temp = [0 for _ in range(size)]
#         return self.count_reverse_pairs(nums, 0, size - 1, temp)

#     def count_reverse_pairs(self, nums, left, right, temp):
#         # 在数组 nums 的区间 [left, right] 统计逆序对
#         if left == right:
#             return 0
#         mid = (left + right) >> 1
#         left_pairs = self.count_reverse_pairs(nums, left, mid, temp)
#         right_pairs = self.count_reverse_pairs(nums, mid + 1, right, temp)

#         reverse_pairs = left_pairs + right_pairs
#         # 代码走到这里的时候，[left, mid] 和 [mid + 1, right] 已经完成了排序并且计算好逆序对

#         if nums[mid] <= nums[mid + 1]:
#             # 此时不用计算横跨两个区间的逆序对，直接返回 reverse_pairs
#             return reverse_pairs

#         reverse_cross_pairs = self.merge_and_count(nums, left, mid, right, temp)
#         return reverse_pairs + reverse_cross_pairs

#     def merge_and_count(self, nums, left, mid, right, temp):
#         """
#         [left, mid] 有序，[mid + 1, right] 有序

#         前：[2, 3, 5, 8]，后：[4, 6, 7, 12]
#         只在后面数组元素出列的时候，数一数前面这个数组还剩下多少个数字，
#         由于"前"数组和"后"数组都有序，
#         此时"前"数组剩下的元素个数 mid - i + 1 就是与"后"数组元素出列的这个元素构成的逆序对个数

#         """
#         for i in range(left, right + 1):
#             temp[i] = nums[i]

#         i = left
#         j = mid + 1
#         res = 0
#         for k in range(left, right + 1):
#             if i > mid:
#                 nums[k] = temp[j]
#                 j += 1
#             elif j > right:
#                 nums[k] = temp[i]
#                 i += 1
#             elif temp[i] <= temp[j]:
#                 # 此时前数组元素出列，不统计逆序对
#                 nums[k] = temp[i]
#                 i += 1
#             else:
#                 # assert temp[i] > temp[j]
#                 # 此时后数组元素出列，统计逆序对，快就快在这里，一次可以统计出一个区间的个数的逆序对
#                 nums[k] = temp[j]
#                 j += 1
#                 # 例：[7, 8, 9][4, 6, 9]，4 与 7 以及 7 后面所有的数都构成逆序对
#                 res += (mid - i + 1)
#         return res

# a = Solution()
# b = a.reversePairs([2,3,5,8,4,6,7,12])
# print(b)

# class Solution(object):
#     def merge(self, nums1, m, nums2, n):
#         """
#         :type nums1: List[int]
#         :type m: int
#         :type nums2: List[int]
#         :type n: int
#         :rtype: void Do not return anything, modify nums1 in-place instead.
#         """
#         # Make a copy of nums1.
#         nums1_copy = nums1[:m] 
#         nums1[:] = []

#         # Two get pointers for nums1_copy and nums2.
#         p1 = 0 
#         p2 = 0
        
#         # Compare elements from nums1_copy and nums2
#         # and add the smallest one into nums1.
#         while p1 < m and p2 < n: 
#             if nums1_copy[p1] < nums2[p2]: 
#                 nums1.append(nums1_copy[p1])
#                 p1 += 1
#             else:
#                 nums1.append(nums2[p2])
#                 p2 += 1

#         # if there are still elements to add
#         if p1 < m: 
#             nums1[p1 + p2:] = nums1_copy[p1:]
#         if p2 < n:
#             nums1[p1 + p2:] = nums2[p2:]
#         return nums1

# class Solution(object):
#     def merge(self, nums1, m, nums2, n):
#         """
#         :type nums1: List[int]
#         :type m: int
#         :type nums2: List[int]
#         :type n: int
#         :rtype: void Do not return anything, modify nums1 in-place instead.
#         """
#         # two get pointers for nums1 and nums2
#         p1 = m - 1
#         p2 = n - 1
#         # set pointer for nums1
#         p = m + n - 1
        
#         # while there are still elements to compare
#         while p1 >= 0 and p2 >= 0:
#             if nums1[p1] < nums2[p2]:
#                 nums1[p] = nums2[p2]
#                 p2 -= 1
#             else:
#                 nums1[p] =  nums1[p1]
#                 p1 -= 1
#             p -= 1
        
#         # add missing elements from nums2
#         nums1[:p2 + 1] = nums2[:p2 + 1]
#         return nums1

# a = Solution()
# b = a.merge([0],0,[1],1)
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

# a = Solution()
# b = a.reversePairs([7,5,6,4])
# print(b)


# class Solution:
#     def addStrings(self, num1: str, num2: str) -> str:
#         res = ""
#         i, j, carry = len(num1) - 1, len(num2) - 1, 0
#         while i >= 0 or j >= 0:
#             n1 = int(num1[i]) if i >= 0 else 0
#             n2 = int(num2[j]) if j >= 0 else 0
#             tmp = n1 + n2 + carry
#             carry = tmp // 10
#             res = str(tmp % 10) + res
#             i, j = i - 1, j - 1
#         return "1" + res if carry else res

# a = Solution()
# b = a.addStrings('123','1789')
# print(b)

# class Solution:
#     def multiply(self, num1, num2):
#         num1_len = len(num1)
#         num2_len = len(num2)
#         res = [0] * (num1_len + num2_len)
#         for i in range(num1_len-1,-1,-1):
#             for j in range(num2_len-1,-1,-1):
#                 tmp = int(num1[i]) * int(num2[j]) + int(res[i+j+1])
#                 res[i+j+1] = tmp%10 # 余数作为当前位
#                 res[i+j] = res[i+j] + tmp//10 # 前一位加上，进位（商作为进位）
#         res = list(map(str, res))
#         # print(res)
#         for i in range(num1_len+num2_len):
#             if res[i]!='0': # 找到第一个非0数字，后面就是结果
#                 return ''.join(res[i:])
#         return '0'

# a = Solution()
# b = a.multiply('123','456')
# print(b)

# class Solution:
#     def trap(self, height):
#         summ = 0
#         if not height: return 0
#         for i in range(1,len(height)-1,1):
#             max_left = 0
#             for j in range(i-1,-1,-1):
#                 if height[j] > max_left:
#                     max_left = height[j]
#             max_right = 0
#             for j in range(i+1, len(height)):
#                 if height[j] > max_right:
#                     max_right = height[j]
                    
#             minn = min(max_left, max_right)
#             if minn > height[i]:
#                 summ += (minn - height[i])
#         return summ

# class Solution:
#     def trap(self, height):
#         summ = 0
#         max_left = 0
#         max_right = 0
#         left = 1
#         right = len(height) - 2
#         for i in range(1,len(height)-1,1):
#             if height[left-1] < height[right+1]:
#                 max_left = max(max_left, height[left-1])
#                 minn = max_left
#                 if minn > height[left]:
#                     summ += (minn - height[left])
#                 left += 1
#             else:
#                 max_right = max(max_right, height[right+1])
#                 minn = max_right
#                 if minn > height[right]:
#                     summ += (minn - height[right])
#                 right -= 1
#         return summ

# class Solution:
#     def trap(self, height):
#         # 边界条件
#         if not height: return 0
#         n = len(height)

#         left,right = 0, n - 1  # 分别位于输入数组的两端
#         maxleft,maxright = height[0],height[n - 1]
#         ans = 0

#         while left <= right:
#             maxleft = max(height[left],maxleft)
#             maxright = max(height[right],maxright)
#             if maxleft < maxright:
#                 ans += maxleft - height[left]
#                 left += 1
#             else:
#                 ans += maxright - height[right]
#                 right -= 1

#         return ans


# a = Solution()
# b = a.trap([0,1,0,2,1,0,1,3,2,1,2,1])
# print(b)

# import heapq


# class MedianFinder:

#     def __init__(self):
#         # 当前大顶堆和小顶堆的元素个数之和
#         self.count = 0
#         self.max_heap = []
#         self.min_heap = []

#     def addNum(self, num):
#         self.count += 1
#         # 因为 Python 中的堆默认是小顶堆，所以要传入一个 tuple，用于比较的元素需是相反数，
#         # 才能模拟出大顶堆的效果
#         heapq.heappush(self.max_heap, (-num, num))
#         _, max_heap_top = heapq.heappop(self.max_heap)
#         heapq.heappush(self.min_heap, max_heap_top)
#         if self.count & 1:
#             min_heap_top = heapq.heappop(self.min_heap)
#             heapq.heappush(self.max_heap, (-min_heap_top, min_heap_top))

#     def findMedian(self):
#         print(self.max_heap)  # 堆排序结果
#         print(self.min_heap)  # 堆排序结果

#         if self.count & 1:
#             # 如果两个堆合起来的元素个数是奇数，数据流的中位数大顶堆的堆顶元素
#             return self.max_heap[0][1]
#         else:
#             # 如果两个堆合起来的元素个数是偶数，数据流的中位数就是各自堆顶元素的平均值
#             return (self.min_heap[0] + self.max_heap[0][1]) / 2

# obj = MedianFinder()
# obj.addNum(1)
# param_2 = obj.findMedian()
# obj.addNum(2)
# param_2 = obj.findMedian()
# obj.addNum(3)
# param_2 = obj.findMedian()
# obj.addNum(4)
# param_2 = obj.findMedian()
# obj.addNum(5)
# param_2 = obj.findMedian()
# obj.addNum(6)
# param_2 = obj.findMedian()
# print(param_2)

# class Solution:
#     def jump(self, nums):
#         end = 0
#         maxPosition = 0
#         steps = 0
#         for i in range(len(nums)-1):
#             maxPosition = max(maxPosition, nums[i] + i)
#             if i == end:
#                 end = maxPosition
#                 steps += 1
#         return steps

# a = Solution()
# b = a.jump([2,3,1,1,4])
# print(b)

# from collections import deque
# class MaxQueue(object):

#     def __init__(self):
#         self.que = deque()
#         self.sort_que = deque()   

#     def max_value(self):
#         """
#         :rtype: int
#         """
#         return self.sort_que[0] if self.sort_que else -1   

#     def push_back(self, value):
#         """
#         :type value: int
#         :rtype: None
#         """
#         self.que.append(value)
#         while self.sort_que and self.sort_que[-1] < value:
#             self.sort_que.pop()
#         self.sort_que.append(value)
        
#     def pop_front(self):
#         """
#         :rtype: int
#         """
#         if not self.que: return -1
#         res = self.que.popleft()
#         if res == self.sort_que[0]:
#             self.sort_que.popleft()
#         return res

# # Your MaxQueue object will be instantiated and called as such:
# obj = MaxQueue()
# param_1 = obj.max_value()
# obj.push_back(1)
# obj.push_back(5)
# obj.push_back(6)
# obj.push_back(4)
# obj.push_back(3)
# obj.push_back(7)
# obj.push_back(2)
# param_2 = obj.max_value()
# print(param_2)
# param_3 = obj.pop_front()
# print(param_3)
# param_4 = obj.pop_front()
# print(param_4)
# param_5 = obj.pop_front()
# print(param_5)
# param_6 = obj.pop_front()
# print(param_6)
# param_7 = obj.pop_front()
# print(param_7)
# param_8 = obj.pop_front()
# print(param_8)

# class Solution:
#     def rotate(self, matrix):
#         """
#         :type matrix: List[List[int]]
#         :rtype: void Do not return anything, modify matrix in-place instead.
#         """
#         n = len(matrix[0])
#         for i in range(n // 2 + n % 2):
#             for j in range(n // 2):
#                 tmp = [0] * 4
#                 row, col = i, j
#                 # store 4 elements in tmp
#                 for k in range(4):
#                     tmp[k] = matrix[row][col]
#                     row, col = col, n - 1 - row
#                 # rotate 4 elements   
#                 for k in range(4):
#                     matrix[row][col] = tmp[(k - 1) % 4]
#                     row, col = col, n - 1 - row
#         return matrix

# class Solution:
#     def rotate(self, matrix):
#         """
#         :type matrix: List[List[int]]
#         :rtype: void Do not return anything, modify matrix in-place instead.
#         """
#         n = len(matrix[0])        
#         for i in range(n // 2 + n % 2):
#             for j in range(n // 2):
#                 tmp = matrix[n - 1 - j][i]
#                 matrix[n - 1 - j][i] = matrix[n - 1 - i][n - j - 1]
#                 matrix[n - 1 - i][n - j - 1] = matrix[j][n - 1 -i]
#                 matrix[j][n - 1 - i] = matrix[i][j]
#                 matrix[i][j] = tmp
#         return matrix

# a = Solution()
# b = a.rotate([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
# print(b)

# 整数拆分
# class Solution(object):
#     def integerBreak(self, n):
#         dp = [1] * (n + 1)
#         for i in range(2, n + 1):
#             for j in range(1, i):
#                 a = i - j
#                 b = dp[i]
#                 c = dp[j]
#                 dp[i] = max(dp[i], max(j, dp[j]) * (i - j))
#         return dp[-1]

# a = Solution()
# b = a.integerBreak(10)
# print(b)

# def partition(arr,low,high): 
#     i = ( low-1 )         # 最小元素索引
#     pivot = arr[high]     

#     for j in range(low , high): 

#         # 当前元素小于或等于 pivot 
#         if   arr[j] <= pivot: 
        
#             i = i+1 
#             arr[i],arr[j] = arr[j],arr[i] 

#     arr[i+1],arr[high] = arr[high],arr[i+1] 
#     return ( i+1 ) 


# arr[] --> 排序数组
# low  --> 起始索引
# high  --> 结束索引

# # 快速排序函数
# def quickSort(arr,low,high): 
#     if low < high: 

#         pi = partition(arr,low,high) 

#         quickSort(arr, low, pi-1) 
#         quickSort(arr, pi+1, high) 

# arr = [1,5,4,6,2,8,7,3,9]
# n = len(arr) 
# quickSort(arr,0,n-1) 
# print ("排序后的数组:") 

# for i in range(n): 
#     print ("%d" %arr[i])

# class Solution:
#     def minimumTotal(self, triangle):
#         """
#         :type triangle: List[List[int]]
#         :rtype: int
#         """
#         res = triangle[-1]
#         for i in range(len(triangle)-2,-1,-1):
#             for j in range(len(triangle[i])):
#                 res[j] = min(res[j],res[j+1]) + triangle[i][j]
#         return res[0]

# class Solution:
#     def minimumTotal(self, triangle):
#         n = len(triangle)
#         f = [0] * n
#         f[0] = triangle[0][0]

#         for i in range(1, n):
#             f[i] = f[i - 1] + triangle[i][i]
#             for j in range(i - 1, 0, -1):
#                 f[j] = min(f[j - 1], f[j]) + triangle[i][j]
#             f[0] += triangle[i][0]
        
#         return min(f)

# class Solution:
#     def minimumTotal(self, triangle):
#         n = len(triangle)
#         f = [[0] * n for _ in range(n)]
#         f[0][0] = triangle[0][0]

#         for i in range(1, n):
#             f[i][0] = f[i - 1][0] + triangle[i][0]
#             for j in range(1, i):
#                 f[i][j] = min(f[i - 1][j - 1], f[i - 1][j]) + triangle[i][j]
#             f[i][i] = f[i - 1][i - 1] + triangle[i][i]
        
#         return min(f[n - 1])

# a = Solution()
# b = a.minimumTotal([[2],[3,4],[6,5,7],[4,1,8,3]])
# print(b)

# class Solution(object):
#     def peakIndexInMountainArray(self, A):
#         lo, hi = 0, len(A) - 1
#         while lo < hi:
#             mi = (lo + hi) // 2
#             if A[mi] < A[mi + 1]:
#                 lo = mi + 1
#             else:
#                 hi = mi
#         return lo

# a = Solution()
# b = a.peakIndexInMountainArray([1,2,3,4,5,6,1])
# print(b)

# import collections

# class Solution:
#     def isBipartite(self, graph):
#         n = len(graph)
#         UNCOLORED, RED, GREEN = 0, 1, 2
#         color = [UNCOLORED] * n
        
#         for i in range(n):
#             if color[i] == UNCOLORED:
#                 q = collections.deque([i])
#                 color[i] = RED
#                 while q:
#                     node = q.popleft()
#                     cNei = (GREEN if color[node] == RED else RED)
#                     for neighbor in graph[node]:
#                         if color[neighbor] == UNCOLORED:
#                             q.append(neighbor)
#                             color[neighbor] = cNei
#                         elif color[neighbor] != cNei:
#                             return False

#         return True

# a = Solution()
# b = a.isBipartite([[1,3],[0,2],[1,3],[0,2]])
# print(b)