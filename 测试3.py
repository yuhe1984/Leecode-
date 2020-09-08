# def merge_sort(nums):
#     if len(nums) <= 1:
#         return nums
#     mid = len(nums) // 2
#     # 分
#     left = merge_sort(nums[:mid])
#     right = merge_sort(nums[mid:])
#     # 合并
#     return merge(left, right)


# def merge(left, right):
#     res = []
#     i = 0
#     j = 0
#     while i < len(left) and j < len(right):
#         if left[i] <= right[j]:
#             res.append(left[i])
#             i += 1
#         else:
#             res.append(right[j])
#             j += 1
#     res += left[i:]
#     res += right[j:]
#     return res

# b = merge_sort([6,5,3,1,8,7,2,4])
# print(b)

# class Solution:
#     def sortArray(self, nums):

#         def merge_sort(nums):
#             if len(nums) <= 1:
#                 return nums
#             mid = len(nums)//2
#             left = merge_sort(nums[:mid])
#             right = merge_sort(nums[mid:])
#             return merge(left, right)

#         def merge(left, right):
#             res = []
#             i = 0
#             j = 0
#             while i < len(left) and j < len(right):
#                 if left[i] >= right[j]:
#                     res.append(right[j])
#                     j += 1
#                 else:
#                     res.append(left[i])
#                     i += 1
#             res += left[i:]
#             res += right[j:]
#             return res

#         nums = merge_sort(nums)
#         return nums

# a = Solution()
# b = a.sortArray([6,5,3,1,8,7,2,4])
# print(b)

# class Solution:
#     def insert(self, intervals, newInterval):
#         intervals = sorted(intervals + [newInterval])
#         tmp = intervals[0]
#         for i in range(1, len(intervals)):
#             if intervals[i][0] <= tmp[-1]:
#                 if intervals[i][1] <= tmp[-1]:
#                     continue
#                 else:
#                     tmp[-1] = intervals[i][1]
#             else:
#                 tmp.extend(intervals[i])
#         return [[tmp[i], tmp[i+1]] for i in range(0, len(tmp), 2)]

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
# b = a.insert([[1,2],[3,5],[6,7],[8,10],[12,16]], [4,8])
# print(b)

# class Solution(object):
#     def minNumber(self, nums):
#         if not nums: return None
#         nums = list(map(str, nums))
#         for i in range(len(nums)):
#             for j in range(i+1, len(nums)):
#                 if (nums[i] + nums[j]) > (nums[j] + nums[i]):
#                     nums[i], nums[j] = nums[j], nums[i]
#         return ''.join(nums)

# class cmp(str):
#     def __lt__(self, other):
#         return self+other < other+self
        
# class Solution:
#     def minNumber(self, nums):
#         nums = sorted( [str(i) for i in nums] , key=cmp)
#         return "".join(nums)

# a = Solution()
# b = a.minNumber([30, 3, 2, 521, 520, 8, 7, 1])
# print(b)

# class Solution:
#     def validateStackSequences(self, pushed, popped):
#         stack = []
#         while pushed:
#             stack.append(pushed.pop(0))
#             while stack and popped and stack[-1] == popped[0]:
#                 popped.pop(0)
#                 stack.pop()
#         return False if stack else True

# a = Solution()
# b = a.validateStackSequences([1,2,3,4,5],[4,5,3,2,1])
# print(b)

# class Solution:
#     def maxSlidingWindow(self, nums, k):
#         deque = [];result = [] # deque也可以用collection里的双端队列实现
#         for i in range(0, len(nums)):
#             while deque and nums[i]>nums[deque[-1]]: # 只存有可能成为最大值的数字的index进deque
#                 deque.pop()
#             deque.append(i)
#             while i-deque[0]>k-1: # 如果相距超过窗口k长度则弃掉
#                 deque.pop(0)
#             if i >= k-1:
#                 result.append(nums[deque[0]]) # 这过程中始终保持deque[0]为最大值的index
#         return result

# a = Solution()
# b = a.maxSlidingWindow([1,2,3,5,4,3,5,6,3,8,9,6,2,3,1],6)
# print(b)

class Solution:
    def swap(self, arr, i, j):
        arr[i], arr[j] = arr[j], arr[i]
    def heapify(self, tree, n, i):
        c1 = 2 * i + 1
        c2 = 2 * i + 2
        if i >= n:
            return
        maxx = i
        if c1 < n and tree[c1] > tree[maxx]:
            maxx = c1
        if c2 < n and tree[c2] > tree[maxx]:
            maxx = c2
        if maxx != i:
            self.swap(tree, maxx, i)
            self.heapify(tree, n, maxx)
    def build_heap(self, tree, n):
        last_node = n - 1
        parent = (last_node - 1) // 2
        for i in range(parent, -1, -1):
            self.heapify(tree, n, i)
    def heap_sort(self, tree, n):
        self.build_heap(tree, n)
        for i in range(n-1, -1, -1):
            self.swap(tree, i, 0)
            self.heapify(tree, i, 0)
    def sortArray(self, nums):
        self.heap_sort(nums,  len(nums))
        return nums

a = Solution()
b = a.sortArray([2, 5, 3, 1, 10, 4])
print(b)