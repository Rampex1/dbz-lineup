import pandas as pd

# Replace with your sheet ID
sheet_id = "1SZMIH8hu3vMTnZfVpRkCJLnR-Jsbr8JMiNn0L1MRWoI"
sheet_gid = "0"  # this is the tab (gid) you want

# Correct export URL
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={sheet_gid}"

# Load the sheet into a DataFrame
df = pd.read_csv(url)

# Save to CSV
df.to_csv("roster.csv", index=False)

print("CSV file created successfully!")
