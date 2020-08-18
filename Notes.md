[TOC]

## Linked-List

### 206. 反转链表

Code

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        # iteration
        prev = None
        while head:
            tmp = head.next
            head.next = prev
            prev = head
            head = tmp
        return prev

        # recursion
        if not head or not head.next:
            return head
        new_head = self.reverseList(head.next)
        head.next.next = head # a tricky step
        head.next = None
        return new_head
```



### 328 奇偶链表

给定一个单链表，把所有的奇数序号节点和偶数序号节点分别排在一起。尝试使用原地算法完成。你的算法的空间复杂度应为 O(1)，时间复杂度应为 O(nodes)，nodes 为节点总数。

```
输入: 2->1->3->5->6->4->7->NULL 
输出: 2->3->6->7->1->5->4->NULL
```

Solution

用两个节点记录奇偶链表各自的尾节点，遇到新节点时分配，最后将奇链表尾连至偶链表头

Code

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        if head is None:
            return None
        odd = head
        even = head.next
        evenHead = head.next
        while even and even.next:
            odd.next = even.next
            odd = odd.next
            even.next = odd.next
            even = odd.next
        odd.next = evenHead
        return head
```

### 面试题 02.08 环路检测/142. 环形链表 II

给定一个链表，实现一个算法返回环路的开头节点。若无，返回None。

Solution

1. 用快慢指针的首次相遇证明有环，记起点到开头节点距离为a，开头节点到首次相遇点距离为b，环路长度为r，快指针绕环n次，则有 ``2*(a+b)=a+b+r*n`` => ``a+b=r*n``
2. For x≥a, we have ``pos(x)=pos(x+r*m)`` for any m. Hence``pos(0+a)=pos(a+r*n)=pos(a+b+a)``. 即从起点和首次相遇点同时出发（走a步），两者会在开头节点相遇。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return None
        slow, fast = head, head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast is slow:
                break
        if not fast or not fast.next:
            return None
        while head is not slow:
            head = head.next
            slow = slow.next
        return head
```

### 138. 复制带随机指针的链表

给定一个链表，每个节点包含一个额外增加的随机指针，该指针可以指向链表中的任何节点或空节点。要求返回这个链表的深拷贝。-10000 <= Node.val <= 10000。`Node.random` 为空（null）或指向链表中的节点。节点数目不超过 1000 。![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/01/09/e1.png)

Solution

- 三次遍历

  1. 第一次遍历在每个原节点后 copy 一个节点

  2. 第二次遍历将每个新加节点的 random 往后移动一位（使之指向具有相同值的新节点）

  3. 第三次遍历将新节点从中分离出来

  - 时间O(n)，空间O(1)【时间3n】

- 利用哈希表的二次遍历

  1. 第一次遍历，向哈希表（dict）中添加“旧节点->新节点”的映射（新节点只保留值，指针置空！）

  2. 第二次遍历，让新节点指针指向“旧节点指针的hash值”

  - 时间O(n)，空间O(n)【时间2n】

Code

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution:
    # Solution 1
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head:
            return None
        # shallow copy
        cur = head
        while cur != None:
            temp = Node(cur.val, cur.next, cur.random)
            cur.next = temp
            cur = cur.next.next
        # move random pointer one step backforwards
        cur = head
        while cur != None:
            if cur.next.random != None:
                cur.next.random = cur.next.random.next
            cur = cur.next.next
        # separate two list
        cur = head.next
        while cur.next:
            cur.next = cur.next.next
            cur = cur.next
        return head.next
    # Solution 2
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head:
            return None
        cur = head
        d = {}
        while cur:
            d[cur] = Node(cur.val, None, None)
            cur = cur.next
        cur = head
        while cur:
            if cur.next:
                d[cur].next = d[cur.next]
            if cur.random:
                d[cur].random = d[cur.random]
            cur = cur.next
        return d[head]
```

### 面试题 02.07. 链表相交

给定两个（单向）链表，判定它们是否相交并返回交点。如果两个链表没有交点，返回 `null` 。可假定整个链表结构中没有循环。

Solution

- 对齐后遍历
  - 分别遍历链表计算长度，让长链表向前走直至补齐差值，遍历对比
- 双指针单次遍历（学习思路）
  - 假设 headA 到交点距离为 a，headB 到交点距离为 b，交点到尾节点距离为 c，有``(a+c)+b=(b+c)+a``即两个指针遍历至末尾时，若从对方头节点继续遍历，则二者会在交点相遇
  - <img src="https://pic.leetcode-cn.com/bb47e810087820bff49a867c8a8de0dfb32a15147bd98b16e8fd93e81d15da31-L.png" alt="L.png" style="zoom:25%;" />

Code

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # Solution 1
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if not headA or not headB:
            return None
        lenA = 0
        lenB = 0
        cur = headA
        while cur:
            lenA += 1
            cur = cur.next
        cur = headB
        while cur:
            lenB += 1
            cur = cur.next
        if lenA > lenB:
            for _ in range(lenA-lenB):
                headA = headA.next
        elif lenB > lenA:
            for _ in range(lenB-lenA):
                headB = headB.next
        while headA is not headB:
            headA = headA.next
            headB = headB.next
        return headA
    # Solution 2
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        p1, p2 = headA, headB
        while p1 or p2:
            if p1 is p2:
                return p1
            p1 = p1.next if p1 else headB
            p2 = p2.next if p2 else headA
        return None
```

### 1171. 从链表中删去总和值为零的连续节点

给定一个链表的头节点 head，要求反复删去链表中由和为 0 的连续节点组成的序列，直到不存在这样的序列为止。删除完毕后，请你返回最终结果链表的头节点。你可以返回任何满足题目要求的答案。

Solution

- 两次遍历
  - 第一次统计累加和，并将``累加和->最后获得此和的节点`` 这个映射放入字典
  - 第二次统计累加和，让当前节点跨越相同累加和的子序列
  - 【假设节点A,B累加和相等，则节点序列[A+1, B]总和值为零】

Code

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def removeZeroSumSublists(self, head: ListNode) -> ListNode:
        dummy = ListNode(0)
        dummy.next = head
        d = {0:dummy}
        cur = head
        cnt = 0
        while cur:
            cnt += cur.val
            d[cnt] = cur
            cur = cur.next
        cur = dummy
        cnt = 0
        while cur:
            cnt += cur.val
            if cnt in d:
                cur.next = d[cnt].next
            cur = cur.next
        return dummy.next
```

### 114. 二叉树展开为链表

给定一个二叉树，原地将它展开为一个单链表

```
例如，给定二叉树
    1
   / \
  2   5
 / \   \
3   4   6
将其展开为：
1
 \
  2
   \
    3
     \
      4
       \
        5
         \
          6
```

Solution

- 将当前右子树放到当前左子树最右节点的右子树上，将当前左子树改为右子树，时间O(n)，空间O(1)

Code

```python
class Solution:
    def flatten(self, root: TreeNode) -> None:
        def get_right_end(node):
            while node.right:
                node = node.right
            return node

        cur = root
        while cur:
            if cur.left:
                right_end = get_right_end(cur.left)
                right_end.right = cur.right
                cur.right = cur.left
                cur.left = None
            cur = cur.right
        return root
```

引申：Morris 遍历算法（时间O(n)，空间O(1)遍历二叉树），见94. 二叉树的中序遍历

### 25. K 个一组反转链表

给你一个链表，每 *k* 个节点一组进行翻转，请你返回翻转后的链表。如果节点总数不是 *k* 的整数倍，那么请将最后剩余的节点保持原有顺序。要求实际交换节点而不是改变内部的值。

Solution

- 检查剩余节点是否有K个/先扫描链表获得长度，反转

Code

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        def reverseNth(head, n):
            cur = head
            prev = None
            for i in range(n):
                post = cur.next
                cur.next = prev
                prev = cur
                cur = post
            head.next = cur
            return prev, head
        cur = head
        prev = None
        new_head = head
        while True:
            # check reversibility
            temp = cur
            for _ in range(k):
                if not temp:
                    return new_head
                temp = temp.next
            cur, old_head = reverseNth(cur,k) # new/old head of sub-sequence
            if new_head is head and k != 1:
                new_head = cur
            if prev:
                prev.next = cur
            prev = old_head
            cur = old_head.next
        return new_head
```

### 82. 删除排序列表中的重复元素 Ⅱ

Solution

- 判断是否重复数字的开端，若是，连续删除具有相同值的节点

Code

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        dummy = ListNode(-1)
        dummy.next = head
        cur = dummy
        while cur.next and cur.next.next:
            if cur.next.val == cur.next.next.val: # check if cur.next is the beginning of deletion
                temp = cur.next
                while temp and temp.val == cur.next.val:
                    temp = temp.next
                cur.next = temp
            else:
                cur = cur.next
        return dummy.next
```

### 1367. 二叉树中的列表

给你一棵以 root 为根的二叉树和一个 head 为第一个节点的链表。如果在二叉树中，存在一条一直向下的路径，且每个点的数值恰好一一对应以 head 为首的链表中每个节点的值，那么请你返回 True ，否则返回 False 。

Solution

- 对于每个子问题中的 root，先判断是否能从 root 开始对应链表中的值，再判断左右子节点为根时是否对应

Code

```python
class Solution:
    def isPartialSubPath(self, head: ListNode, root: TreeNode) -> bool:
        if not head:
            return True
        elif not root:
            return False
        if head.val == root.val:
            return self.isPartialSubPath(head.next, root.left) or \
        		self.isPartialSubPath(head.next, root.right)
        else:
            return False

    def isSubPath(self, head: ListNode, root: TreeNode) -> bool:
        if not head:
            return True
        elif not root:
            return False
        if head.val == root.val:
            return self.isPartialSubPath(head.next, root.left) or \
                self.isPartialSubPath(head.next, root.right) or \
                self.isSubPath(head, root.left) or \
                self.isSubPath(head, root.right)
        else:
            return self.isSubPath(head, root.left) or \
                    self.isSubPath(head, root.right)
```

### 86. 分隔链表

给定一个链表和一个特定值 *x*，对链表进行分隔，使得所有小于 *x* 的节点都在大于或等于 *x* 的节点之前。保留两个分区中每个节点的初始相对位置。

**示例:**

```
输入: head = 1->4->3->2->5->2, x = 3
输出: 1->2->2->4->3->5
```

Solution

- 用两个指针分别维护一个子链表，最后将二者连接起来

Code

```python
class Solution:
    def partition(self, head: ListNode, x: int) -> ListNode:
        small = ListNode(None)
        large = ListNode(None)
        small_head = small
        large_head = large
        while head:
            if head.val < x:
                small.next = head
                small = small.next
            else:
                large.next = head
                large = large.next
            head = head.next
        large.next = None
        small.next = large_head.next
        return small_head.next
```

### 面试题 02.06. 回文链表

检查输入的链表是否回文

Solution

- 快慢指针找到中点后，反转后半链表，最后和前半链表作比较

Code

```python
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        if not head:
            return True
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        pre, cur, post = None, slow, slow.next
        while cur:
            cur.next = pre
            pre = cur
            cur = post
            if post:
                post = post.next
        # pre is the head of right-half linked-list
        # if len(head) is odd, the end of pre and head should be the same node
        while pre: 
            if pre.val != head.val:
                return False
            pre = pre.next
            head = head.next
        return True
```

### 445. 两数相加 II

给定两个非空链表来代表两个非负整数。数字最高位位于链表开始位置。它们的每个节点只存储一位数字。将这两数相加会返回一个新的链表。可以假设除了数字 0 之外，这两个数字都不会以零开头。

**示例：**

```
输入：(7 -> 2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 8 -> 0 -> 7
```

Solution

- 用栈（list）存储值，倒序构建新的链表

Code

```python
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1: return l2
        if not l2: return l1
        s1, s2 = [], []
        while l1:
            s1.append(l1.val)
            l1 = l1.next
        while l2:
            s2.append(l2.val)
            l2 = l2.next
        cur = None
        remaining = 0
        while s1 or s2 or remaining > 0:
            digit_sum = remaining
            if s1:
                digit_sum += s1.pop()
            if s2:
                digit_sum += s2.pop()
            new_node = ListNode(digit_sum%10)
            remaining = digit_sum // 10
            new_node.next = cur
            cur = new_node
        return cur
```

### 143. 重排链表

给定一个单链表 *L*：*L*0→*L*1→…→*L*n-1→*L*n ，将其重新排列后变为： *L*0→*L*n→*L*1→*L*n-1→*L*2→*L*n-2→… 要求进行实际节点交换而不是单纯改变节点内部值。

Solution

- 方法一
  - 
- 方法二
  - 快慢指针找到中位节点后翻转后半节点，将后半节点整合进前半

Code

```python
class Solution:
    # Solution 1
    def reorderList(self, head: ListNode) -> None:
        
    # Solution 2
    def reorderList(self, head: ListNode) -> None:
        if not head or not head.next:
            return head
        # let slow stop at or before the median so that we can set the end of first half to be None
        slow, fast = head, head.next 
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        tmp = slow
        slow = slow.next
        tmp.next = None
        prev, cur = None, slow
        while cur:
            next_node = cur.next
            cur.next = prev
            prev = cur
            cur = next_node
        head1, head2 = head, prev
        while head2:
            next_head1 = head1.next
            head1.next = head2
            head2 = head2.next
            head1.next.next = next_head1
            head1 = next_head1
        if head1:
            head1.next = None
        return head
```



## Array

### 001 两数之和

给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。

```
Input: 
[2, 7, 11, 15]
9
Output: 
[0,1]
```

Solution

- 暴力解

  - 为每个值遍历一遍数组，时间复杂度O(n^2)，空间复杂度O(1)

- 哈希法

  1. 为每个整数-序号的键值对构造映射，在key中搜索另一个加数，时间复杂度O(n)，空间复杂度O(n)

  2. item in dict 平均时间复杂度O(1) 【基础知识.md/Python/数据结构/dict】

Code

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        a = {}
        for i, val in enumerate(nums):
            if a.get(target-val) is not None:
                return [a.get(target-val), i]
            a[val] = i
```



## Tree

### 94. 二叉树的中序遍历

给定一个二叉树，返回它的*中序* 遍历。

Solution

- 方法一，递归（时间O(n)，空间O(logn)）
- 方法二，[Morris 遍历算法](https://www.cnblogs.com/anniekim/archive/2013/06/15/morristraversal.html)（时间O(n)，空间O(1)）
  - 若当前节点的左子树不存在，则输出当前节点，右节点成为新的当前节点
  - 若存在
    - 若中序遍历的前驱节点的右子节点（左一下，一直往右）为空，则该右子节点指向当前节点，本左子节点成为新的当前节点；
    - 若为当前节点，则将其置空（还原），输出当前节点，本右子节点成为新的当前节点
  - 看上面链接的复杂度分析为什么Morris遍历时间复杂度是O(n)

Code

```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        # recursion
        if not root:
            return []
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)

        # morris traveral
        cur = root
        res = []
        while cur:
            if not cur.left:
                res.append(cur.val)
                cur = cur.right
            else:
                prev = cur.left
                while prev.right and prev.right is not cur:
                    prev = prev.right
                if not prev.right:
                    prev.right = cur
                    cur = cur.left
                else:
                    prev.right = None
                    res.append(cur.val)
                    cur = cur.right
        return res
```



## Number

### 343. 整数拆分

给定一个正整数 *n*，将其拆分为至少两个正整数的和，并使这些整数的乘积最大化。 返回你可以获得的最大乘积。

Solution
$$
设 g(t)=e^{t},h(x)=\frac{\ln x}{x},f(x)=x^{\frac{n}{x}}，则有f(x)=e^{\frac{n\ln x}{x}}=g(n\cdot h(x))
\\
\because g(t)在t\in [0,+\infty)上单调递增
\therefore h(x)和f(x)单调性相同
\\
f(x)极大值在h'(x)=\frac{1-\ln x}{x^2}=0 即x=e处取得（证明略）
\\
\frac{f(2)}{f(3)}=e^{n(\frac{\ln 2}{2}-\frac{\ln 3}{3})}=e^{\frac{n}{6}(\ln 8-\ln 9)}<1
\\
f(x)在x=3处取得最大值，应将n尽可能拆分成3的和，并分类讨论n\%3!=0时的处理
$$
Code

```python
class Solution:
    def integerBreak(self, n: int) -> int:
        if n < 4:
            return n-1
        if n%3 == 0:
            return pow(3,n//3)
        elif n%3 == 1:
            return pow(3,n//3-1)*4
        elif n%3 == 2:
            return pow(3,n//3)*2
```

### 面试题 08.03 魔术索引

魔术索引。 在数组A[0...n-1]中，有所谓的魔术索引，满足条件A[i] = i。给定一个有序整数数组，编写一种方法找出魔术索引，若有的话，在数组A中找出一个魔术索引，如果没有，则返回-1。若有多个魔术索引，返回索引值最小的一个。

Code

```python
class Solution:
    def findMagicIndex(self, nums: List[int]) -> int:
        # Solution 1
        for i in range(len(nums)):
            if i == nums[i]:
                return i
        return -1
    	# Solution 2
        n = 0
        while n < len(nums):
            if n == nums[n]:
                return n
            elif n < nums[n]:
                n = nums[n]
            else:
                n += 1
        return -1
```

