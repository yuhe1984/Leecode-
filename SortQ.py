class Solution:
    def sort(self, nums, i, j):
        l = i - 1
        for k in range(len(nums)):
            if nums[k] < nums[j]:
                i = l + 1
                nums[i], nums[j] = nums[j], nums[i]
        i += 1
        nums[i], nums[j] = nums[j], nums[i]
        return nums
                
    def sortP(self, nums, p, q):
        mid = (p + q) // 2
        while p < q:
            self.sortP(nums, p, mid)
            self.sortP(nums, mid, q)
        return self.sort(nums, 0, len(nums)-1)
        
    def SortQ(self, nums):
        self.sortP(nums, 0, len(nums)-1)
        return nums

a = Solution()
b = a.SortQ([2,1,5,3])
print(b)