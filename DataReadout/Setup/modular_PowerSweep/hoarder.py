import runMLCode as rmc
import numpy as np
import os
import glob

def hoarder():

    # *** unfinished ***
    # goes through MKID_DATA_DIR and finds and tests training data

    
    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

    # load trained model
    mlClass = rmc.get_power_NN()

    f = []
    dparray = []
    fnarray = []
    for (dirpath, _, filenames) in walk(mdd):
        dparray.append(dirpath)
        fnarray.append(filenames)

    # loop through all directories
    for day in dparray[1:18]:
        
        # ignore those after Hal was created for now
        date = day.split('/')[-1][:8] #< 22000000

        try:
            int(date)
        except ValueError:
            date = day.split('/')[-2][:8]
        print day, date
        if date != '':

            # print date
            # try:
            if int(date) < 20160600:
                os.chdir(day)

                # locate the h5 files
                h5files = glob.glob("ps*.h5")
                # print h5files
                if h5files != []:
                    for ps in h5files:

                            # does it have manual click throughs
                            basefile = ('.').join(ps.split('.')[:-1])
                            PSFile = day+'/'+basefile[:-16] + '.txt'                               
                            if os.path.isfile(PSFile):
                                print 'found corresponding ps click through'


                                # h5File = rawTrainFiles[1]
                                # h5File = os.path.join(mdd,h5File)
                                # PSFile = h5File[:-19] + 'man_agreed.txt' #'x-reduced.txt'
                                # mlClass = get_power_NN(PSFile)

                                print ps, PSFile

                                # evaluate on this dataset
                                matches = mlt.evaluateModel(mlClass, h5file = ps)
                                print matches
                                exit()

                                # if score is above certain level train on it
                                # rawTrainFiles.append(day+h5file)
                                # mlClass = get_power_NN(h5file = ps, PSFile=PSFile)


                            else:
                                print 'no ps click-through with conventional name'

                                txtfiles = glob.glob("ps*.txt")
                                similarity = np.zeros((len(txtfiles)))
                                for it, tf in enumerate(txtfiles):
                                    similarity[it] = similar(basefile, tf)
                                # print similarity
                                # print np.argmax(similarity)
                                print 'most similar name: ', txtfiles[np.argmax(similarity)]

                                print ps, PSFile

                                # check contents with PSFitMLData
                                
                                # evaluate on this dataset
                                try:
                                    matches = mlt.evaluateModel(mlClass, initialFile = ps)
                                    print len(matches)
                                    print matches

                                    # if score is above certain level train on it
                                    # rawTrainFiles.append(day+h5file)
                                    mlClass = rmc.get_power_NN(h5file = ps, PSFile=PSFile)

                                except Exception:
                                    print '\nCannot train on file: ', txtfiles[np.argmax(similarity)]


            # except TypeError:
            #     print 'folder name not in a recognisable format'
            # # if int(day.split('/')) > 20160600:
            # #     break

    # list of usefable training data goes here
    return 1