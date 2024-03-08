# query_name="Q6CB74"
# awk -v query="$query_name" '$1 ~ query || $2 ~ query {print $3}' alipid.txt

# import subprocess

# # Define the query name
# query_name = "Q6CB74"

# # Define the shell command
# command = f'awk -v query="{query_name}" \'$1 ~ query || $2 ~ query {{print $3}}\' alipid.txt | sort -nr | head -n 1'

# # Run the shell command
# process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
# output, error = process.communicate()

# # Convert the output to a float
# max_id = float(output.decode('utf-8').strip())

# # Print the maximum %id
# print(max_id)

'''
1. Select query
2. Insert query into train
3. phmmer query against train
4. If query maxpid < 0.25, then insert query into test
5. remove query from train
6. Else repeat 2-5 with test

Details for step 4.
phmmer query with incE of 0.001 against train/test (should include query) and output to alignment
esl-alipid the resulting alignment
awk to get the max %id
'''