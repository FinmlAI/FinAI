import pandas as pd

def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

flattened_data = [flatten_json(record) for record in data_list]

print(f"Flattened {len(flattened_data)} records.")
# Display the first flattened record to show the structure
if flattened_data:
    print("\nExample of a flattened record:")
    print(json.dumps(flattened_data[0], indent=2))