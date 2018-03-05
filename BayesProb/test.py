def groupAnagrams(strs):
    strMap = {}
    result = []
    for str in strs:
        target = "".join(sorted(str))
        if target not in strMap:
            strMap[target] = [str]
        else:
            strMap[target].append(str)
    for value in strMap.values():
        result += [sorted(value)]
    return sorted(result)

if __name__ == "__main__":
    strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
    result = groupAnagrams(strs)
    print(result)

