with open('results.txt', 'r') as infile, \
    open('results2.txt', 'w') as outfile:

    line = infile.readline()
    for l in line.split('acc='):
        if len(l) == 0:
            continue
        outfile.write(
            'acc=' + l + '\n'
        )