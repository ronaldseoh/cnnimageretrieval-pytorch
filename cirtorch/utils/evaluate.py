import numpy as np
import wandb

def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    
    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images
    
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap

def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.

         Usage: 
           map = compute_map (ranks, gnd) 
                 computes mean average precsion (map) only
        
           map, aps, pr, prs = compute_map (ranks, gnd, kappas) 
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
        
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd) # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos  = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgndj)]

        k = 0;
        ij = 0;
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1 # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j]); 
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs


def compute_map_and_print(dataset, ranks, gnd, kappas=[1, 5, 10], wandb_enabled=False, epoch=-1, global_step=-1):
    
    # old evaluation protocol
    if dataset.startswith('oxford5k') or dataset.startswith('paris6k'):
        map, aps, _, _ = compute_map(ranks, gnd)
        
        if wandb_enabled:
            wandb.log({"test_map_" + dataset: map, "epoch": epoch, "global_step": global_step})
            wandb.log({"test_aps_" + dataset: aps, "epoch": epoch, "global_step": global_step})

        print('>> {}: mAP {:.2f}'.format(dataset, np.around(map*100, decimals=2)))

    # new evaluation protocol
    elif dataset.startswith('roxford5k') or dataset.startswith('rparis6k'):
        
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy']])
            g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
            gnd_t.append(g)

        mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, kappas)

        if wandb_enabled:
            wandb.log({"test_mapE_" + dataset: mapE, "epoch": epoch, "global_step": global_step})
 
            for i in np.arange(len(gnd_t)):
                wandb.log({"test_apsE_" + str(i) + '_' + dataset: apsE[i], "epoch": epoch, "global_step": global_step})

            wandb.log({"test_mprE_" + dataset: mprE, "epoch": epoch, "global_step": global_step})
            
            for i in np.arange(len(gnd_t)):
                for j in np.arange(len(kappas)):
                    wandb.log({"test_prsE_" + str(i) + '_' + str(j) +'_' + dataset: prsE[i, j], "epoch": epoch, "global_step": global_step})

        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk']])
            gnd_t.append(g)

        mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, kappas)
        
        if wandb_enabled:
            wandb.log({"test_mapM_" + dataset: mapM, "epoch": epoch, "global_step": global_step})

            for i in np.arange(len(gnd_t)):
                wandb.log({"test_apsM_" + str(i) + '_' + dataset: apsM[i], "epoch": epoch, "global_step": global_step})

            wandb.log({"test_mprM_" + dataset: mprM, "epoch": epoch, "global_step": global_step})

            for i in np.arange(len(gnd_t)):
                for j in np.arange(len(kappas)):
                    wandb.log({"test_prsM_" + str(i) + '_' + str(j) +'_' + dataset: prsM[i, j], "epoch": epoch, "global_step": global_step})

        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
            gnd_t.append(g)

        mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, kappas)
        
        if wandb_enabled:
            wandb.log({"test_mapH_" + dataset: mapH, "epoch": epoch, "global_step": global_step})

            for i in np.arange(len(gnd_t)):
                wandb.log({"test_apsH_" + str(i) + '_' + dataset: apsH[i], "epoch": epoch, "global_step": global_step})

            wandb.log({"test_mprH_" + dataset: mprH, "epoch": epoch, "global_step": global_step})

            for i in np.arange(len(gnd_t)):
                for j in np.arange(len(kappas)):
                    wandb.log({"test_prsH_" + str(i) + '_' + str(j) +'_' + dataset: prsH[i, j], "epoch": epoch, "global_step": global_step})

        print('>> {}: mAP E: {}, M: {}, H: {}'.format(dataset, np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
        print('>> {}: mP@k{} E: {}, M: {}, H: {}'.format(dataset, kappas, np.around(mprE*100, decimals=2), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))
