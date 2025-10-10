pos = 0

# responsible for constantly polling for any updates to csv file
def reader(file, encoding = "utf-8"):
    global pos
    # file is read in binary mode, so that:
    # file pointer can be modified and referenced
    file.seek(pos)
    for line in file:
        if line.strip():
            yield line.decode("utf-8")

    if (pos != file.tell()):
        pos = file.tell()