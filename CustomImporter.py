import glob
import os



def OMR_Importer(files):
    
    Datafiles=[sorted(glob.glob(f'../input/12-hand-gestures/{file}' + '/*.npy')) for file in files]
    

    #Data for video
    actionsO=np.array(np.concatenate([load(x).reshape(20,30,48,64,3)[:19]/255 for x in Datafiles[0]],axis=0))
    actionsO=np.concatenate([actionsO*3])
    
    #Data for skeleton 
    actionsML=np.array(np.concatenate([load(x).reshape(20,30,126)[:19,:,0:63]*(-1) for x in Datafiles[1]],axis=0))
    actionsMR=np.array(np.concatenate([load(x).reshape(20,30,126)[:19,:,63:126] for x in Datafiles[1]],axis=0))
    actionsM=np.concatenate([actionsML,actionsMR],axis=2)
    actionsM=np.concatenate([actionsM,actionsM+0.5,actionsM-0.5],axis=0)
    
    #Data for Radar 
    actionsRXY=np.array(np.concatenate([load(x).reshape(20,30,1000)[:19,:,0:400] for x in Datafiles[2]],axis=0))
    actionsRVR=np.array(np.concatenate([load(x).reshape(20,30,1000)[:19,:,600:1000] for x in Datafiles[2]],axis=0))
    actionsR=np.concatenate([actionsRXY,actionsRVR],axis=2)
    actionsR=np.concatenate([actionsR,actionsR+0.5,actionsR-0.5],axis=0)

    
    actions=[]
    for x in sorted(os.listdir('../input/12-hand-gestures/MP_OPTICAL_SMALL')):
        if x.endswith(".npy"):
            # Prints only text file present in My Folder
            actions.append(x)
    actions =np.array(actions)
    print (actions)

    label_map = {label:num for num, label in enumerate(actions)}


    # 19 videos worth of data
    no_sequences = 19

    # Videos are going to be 30 frames in length
    sequence_length = 30
    label_map
    print (label_map)
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            labels.append(label_map[action])
            

    y = to_categorical(labels).astype(int)
    y=np.concatenate([y*3])
    labels=np.concatenate([labels*3])
    print(actionsO.shape,actionsM.shape,actionsR.shape,y.shape)
