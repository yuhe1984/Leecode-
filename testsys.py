import sys
'''sys.stdin = open('sys.txt')
k = int(sys.stdin.readline().strip())
data = []
for i in range(k):
    data.append(list(map(int,sys.stdin.readline().strip().split())))

print(k)
print(data)'''

import sys
import bisect
lines = sys.stdin.readlines()
line = lines[0].strip().split()
n = int(line[0])
m = int(line[1])
job = {}
for line in lines[1:-2]:
    line = line.strip().split()
    if not line:
        continue
    job[int(line[0])] = int(line[1])
arr = sorted(job.keys())
for i in range(len(arr)-1):
    if job[arr[i]] > job[arr[i+1]]:
          job[arr[i+1]] = job[arr[i]]
abi = map(int, lines[-1].strip().split(' '))
for i in abi:
    ind = bisect.bisect(arr,i)
    if ind == 0:
        print(0)
    else:
        print(job[arr[ind-1]])
