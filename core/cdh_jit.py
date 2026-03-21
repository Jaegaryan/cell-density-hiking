from numba import njit


@njit(cache=True)
def get_scan_length(n):
    """ Return smalles power of two >= n """
    scan_length = 1
    while scan_length < n:
        scan_length *= 2
    return max(2048, scan_length)  # dispatches have >=1024 threads, spanning >=2028 values (for blelloch scan)


@njit(cache=True)
def parents(edges, peaks, cols):
    n_assigned_parents = 0  # count assigned parents for early stopping
    for edge in edges[::-1]:
        small = edge['from']
        large = edge['to']

        # find smaller peak without parent in peak chains
        while peaks[small]['parent'] != 0xFFFFFFFF:
            small = peaks[small]['parent']
            if peaks[small]['height'] > peaks[large]['height']:
                small, large = large, small

        # check for cycle at same height, assign to next higher parent without parent
        while (peaks[small]['height'] == peaks[large]['height'] and peaks[large]['parent'] != 0xFFFFFFFF):  # will eventually reach highest peak or small
            large = peaks[large]['parent']
        if small != large:  # avoid recurrent parent
            # assign parent
            peaks[small]['parent'] = large
            peaks[small]['prominence'] = peaks[small]['height'] - edge['col_height']
            cols[small] = edge['col']  # for visualization

            # if all but highest peak are assigned don't need to iterate through remaining edges
            n_assigned_parents += 1
            if n_assigned_parents == len(peaks)-1:
                break

    # TODO do for multiple highest peaks when disconneted graph
    # hike up to highest peak to assign max prominence (large at this point is a high peak -> few jumps to highest peak)
    while peaks[large]['parent'] != 0xFFFFFFFF:
        large = peaks[large]['parent']
    peaks[large]['prominence'] = peaks[large]['height']
    peaks[large]['parent'] = large
    cols[large] = large

    return peaks, cols, large  # large = index of highest peak
