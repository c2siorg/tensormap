#!/bin/sh

max_attempts=5
wait_time=5
attempt_num=1
success=0

while [ $attempt_num -le $max_attempts ]
do
  flask db init 
  flask db migrate
  flask db upgrade
  if [ $? -eq 0 ]
  then
    success=1
    break
  else
    echo "Command failed, attempt $attempt_num of $max_attempts. Retrying in $wait_time seconds..."
    sleep $wait_time
    attempt_num=$((attempt_num+1))
  fi
done

if [ $success -eq 1 ]
then
  exec "$@"
else
  echo "Command failed after $max_attempts attempts, stopping."
  exit 1
fi