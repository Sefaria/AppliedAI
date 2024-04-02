#ensure that the Docker image builds correctly and the container runs as expected.
'''
Build the Docker image.
Run the container.
Check the logs for the specific message.
Clean up after tests are done.
'''

import subprocess
import time
import sys

# Settings
image_name = 'your_image_name'  # Replace with your actual image name
container_name = 'test_container'  # Temporary name for the test container
log_check_phrase = 'Bolt app is running'  # The log phrase to check for
max_wait_time = 60  # Maximum number of seconds to wait for the app to start

# Function to execute shell commands
def run_command(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print(f"Command '{' '.join(command)}' failed with the following output:")
        print(result.stdout)
        sys.exit(result.returncode)
    return result.stdout

# Step 1: Build the image
print(f"Building Docker image '{image_name}'...")
run_command(['docker', 'build', '-t', image_name, '.'])

# Step 2: Run the container
print(f"Running container '{container_name}' from image '{image_name}'...")
run_command(['docker', 'run', '--name', container_name, '-d', image_name])

# Step 3: Check logs for the 'Bolt app is running' message
print("Checking logs for the 'Bolt app is running' message...")
start_time = time.time()
while True:
    logs = run_command(['docker', 'logs', container_name])
    if log_check_phrase in logs:
        print("Success: Found the 'Bolt app is running' message in the logs.")
        break
    if time.time() - start_time > max_wait_time:
        print("Failure: Timed out waiting for the 'Bolt app is running' message.")
        break
    time.sleep(5)  # Wait for a few seconds before checking the logs again

# Step 4: Cleanup
print("Cleaning up: Stopping and removing the test container...")
run_command(['docker', 'stop', container_name])
run_command(['docker', 'rm', container_name])

print("Docker tests completed.")
