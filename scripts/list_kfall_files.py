from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

dataset = "usmanabbasi2002/kfall-dataset"
all_names = []
token = ""
while True:
    resp = api.dataset_list_files(dataset, page_token=token, page_size=200)
    batch = [f.name for f in resp.files]
    if not batch:
        break
    all_names.extend(batch)
    print(f"fetched {len(batch)} files (total so far {len(all_names)})")
    token = getattr(resp, "nextPageToken", "") or getattr(resp, "next_page_token", "")
    if not token:
        break

print(f"TOTAL FILES: {len(all_names)}")
sensor_sa06 = [n for n in all_names if "SA06" in n and "sensor_data" in n.lower()]
print(f"SA06 sensor-related files found: {len(sensor_sa06)}")
for n in sensor_sa06[:15]:
    print(" ", n)