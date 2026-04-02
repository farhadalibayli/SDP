employees = {
    1: {"name": "Alice"},
    2: {"name": "Bob"},
    3: {"name": "Charlie"}
}

departments = {
    1: {"department": "IT"},
    2: {"department": "HR"},
    4: {"department": "Finance"}
}

def inner_join_dict(left, right):
    result = {}

    for key in left:
        if key in right:
            result[key] = {**left[key], **right[key]}

    return result