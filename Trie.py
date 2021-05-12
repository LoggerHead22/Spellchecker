
from collections import deque, OrderedDict


class Node:
    def __init__(self, char, flag=0):
        self.flag = flag
        self.freq = None
        self.child_nodes = {}

    def __repr__(self):
        return f'<{int(self.flag)},{self.child_nodes.keys()}>'

    def add_to_node(self, word):
        curr_node = self
        word = list(word[::-1])
        while word != []:
            if word[-1] not in curr_node.child_nodes:
                curr_node.child_nodes[word[-1]] = Node("")
#                curr_node.child_nodes = OrderedDict(
#                    sorted(curr_node.child_nodes.items()))

            curr_node = curr_node.child_nodes[word[-1]]
            word.pop()

        if curr_node.flag:
            curr_node.flag += 1
            result = False
        elif not curr_node.flag:
            curr_node.flag = 1
            result = True

        return result

    def find_in_node(self, word):
        curr_node = self
        result = None
        word = list(word[::-1])
        while word != []:
            if word[-1] not in curr_node.child_nodes:
                result = (False, None)
                break
            else:
                curr_node = curr_node.child_nodes[word[-1]]
            word.pop()

        if result is None:
            result = (curr_node.flag, curr_node)

        return result

    def pop_in_node(self, word):
        if word == "":
            if self.flag and len(self.child_nodes) == 0:
                return True
            elif self.flag and len(self.child_nodes) != 0:
                self.flag = False
                return False
            else:
                raise KeyError
        else:
            if word[0] not in self.child_nodes:
                raise KeyError
            else:
                result = self.child_nodes[word[0]].pop_in_node(word[1:])

                if result:
                    del self.child_nodes[word[0]]
                    result = len(self.child_nodes) == 0

                return result and not self.flag


class TrieIterator:
    def __init__(self, root, prefix=""):
        self.prefix = prefix
        self.root = root
        self.gener_words = TrieIterator.bfs(self) if root is not None else None

    def __iter__(self):
        return self

    def __next__(self):
        if self.gener_words is None:
            raise StopIteration()
        try:
            word = next(self.gener_words)
        except StopIteration:
            raise StopIteration()

        return word

    def bfs(self):
        queue = deque([("", self.root)])
        queue_prefix = deque([["", 1]])

        while queue:
            curr_char, curr_node = queue.popleft()
            curr_prefix = queue_prefix[0][0]

            queue_prefix[0][1] -= 1
            if queue_prefix[0][1] == 0:
                queue_prefix.popleft()

            if curr_node.flag == 1:
                yield self.prefix + curr_prefix + curr_char

            for child in OrderedDict(sorted(curr_node.child_nodes.items())):
                queue.append((child, curr_node.child_nodes[child]))

            if len(curr_node.child_nodes) > 0:
                queue_prefix.append([curr_prefix + curr_char,
                                     len(curr_node.child_nodes)])


class Trie:
    def __init__(self):
        self.root_node = Node("", False)
        self._len = 0
        self.trie_iterator = None

    def add(self, key):
        res = self.root_node.add_to_node(key)
        if res:
            self._len += 1
        return res

    def pop(self, word):
        try:
            self.root_node.pop_in_node(word)
        except KeyError:
            raise KeyError(word)
        else:
            self._len -= 1

    def __len__(self):
        return self._len

    def __contains__(self, word):
        return self.root_node.find_in_node(word)[0]

    def __iter__(self):
        return TrieIterator(self.root_node, "")

    def starts_with(self, prefix):
        start_node = self.root_node.find_in_node(prefix)[1]
        return TrieIterator(start_node, prefix)