jobs | awk '{print $1}' | xargs kubectl  delete job
