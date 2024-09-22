import re

def extract_week(func_name):
    pattern = r'^(week\d+)_lecture$'
    match = re.match(pattern, func_name)
    if match:
        return match.group(1)
    return None

def parse_function_signatures(function_signatures):
    result = []
    for signature in function_signatures:
        # Remove the parentheses and split by the first '('
        func_name, params = signature.split('(', 1)
        # Remove the closing parenthesis and split by ','
        params = params.rstrip(')').split(', ')
        result.append((func_name, params))

    return result