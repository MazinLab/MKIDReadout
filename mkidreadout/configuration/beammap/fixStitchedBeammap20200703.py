from mkidcore.objects import Beammap
import numpy as np

def generate_beammap_stuff():
    beammap = Beammap(file='/mnt/data0/MEC/20200620/stitched_together_map_correction4.txt', xydim=(140,146), default='MEC')

    bm = np.array((beammap.resIDs, beammap.flags, beammap.xCoords, beammap.yCoords)).T

    flnums = [1,2,3,4,5,6,7,8,9,10]

    masks = []
    for i in flnums:
        m = np.floor(bm[:, 0]/10000) == i
        masks.append(m)

    fls = []
    for i in range(len(flnums)):
        fl = bm[masks[i]]
        fls.append(fl)

    return fls, bm, beammap


def generate_fl_stuff(fl, flnum, bm_list):
    bm = bm_list
    bmgrid = np.zeros((146,140))
    for i in fl:
        if (i[3] < 146) and (i[2] < 140) and (i[1] == 0):
            # print(i)
            bmgrid[i[3].astype(int)][i[2].astype(int)] += 1

    doubleCoords = np.array(np.where(bmgrid == 2)).T
    openCoords = np.array(np.where(bmgrid == 0)).T

    xmin = ((flnum - 1) * 14)
    xmax = (flnum * 14) - 1

    flOpen = []
    for i in openCoords:
        if (i[1] >= xmin) and (i[1] <= xmax):
            flOpen.append(i)
    flOpen = np.array(flOpen)

    residsToFix = []
    for i in doubleCoords:
        xm = bm[:, 2] == i[1]
        ym = bm[:, 3] == i[0]
        fm = bm[:, 1] == 0
        mm = xm & ym & fm
        temp = bm[mm]
        r = []
        r.append(i[1])
        r.append(i[0])
        for res in temp:
            r.append(res[0])
        residsToFix.append(r)

    closests = []
    for i in residsToFix:
        c = find_closest_point_to_overlap(i, flOpen)
        closests.append(c)

    return bmgrid, residsToFix, flOpen, doubleCoords, closests

def do_resolution(residsToFix, closests):
    resolved = []
    for i, j in zip(residsToFix, closests):
        resolved.append(resolve_good_coords(i, j))
    return resolved

def find_closest_point_to_overlap(overlapInfo, openCoords):
    dists = []
    for i in openCoords:
        dist = np.sqrt(((overlapInfo[1]-i[0])**2) + ((overlapInfo[0]-i[1])**2))
        dists.append(dist)

    dmask = dists == np.min(dists)
    closestCoords = openCoords[dmask]
    return closestCoords

def resolve_good_coords(resIDs_to_fix, closest_open_neighbors):
    original_coord = np.array((resIDs_to_fix[0], resIDs_to_fix[1]))
    close_coords = closest_open_neighbors

    resIDs = resIDs_to_fix[2:]

    idxToStay = np.random.randint(low=0, high=100) % (len(resIDs))

    newCoords = []
    unmovedResonator = np.array((resIDs[idxToStay], original_coord[0], original_coord[1]))
    newCoords.append(unmovedResonator)

    resIDs.remove(resIDs[idxToStay])
    for i in range(len(resIDs)):
        temp = np.array((resIDs[i], close_coords[i][1], close_coords[i][0]))
        newCoords.append(temp)

    return newCoords

def update_beammap(resolvedResonators, beammap):
    """
    Steps to take in resolved list and beammap object and fix it
    :param resolvedResonators:
    :param beammap:
    :return:
    """

def run_resolver():
    fls, bm, beammap = generate_beammap_stuff()
    all_resolved = []
    for i in [1,2,3,4,5,6,7,8,9,10]:
        bmgrid, residsToFix, flopen, doublecoords, closestpoints = generate_fl_stuff(fls[i-1], i, bm)
        resolved = do_resolution(residsToFix, closestpoints)
        all_resolved.append(resolved)
    return all_resolved
