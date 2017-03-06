from multiprocessing import Process
import Main_structure
import os.path

def main(protein_list,model_type,model_size_flag,num_calibrations,recalibrate):
    if isinstance(protein_list,str):
        protein_list = [protein_list]
    for protein in protein_list:
        p = Process(target=Main_structure.main,
                    kwargs={'target_protein':protein, 'model_size_flag':model_size_flag,
                            'model_testing_list':model_type, 'num_calibrations':num_calibrations,
                            'recalibrate':recalibrate})
        try:
            p.start()
        except KeyboardInterrupt:
            print("searching stopped")
        p.join()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--protein-list',nargs='+')
    parser.add_argument('--model-type',nargs='+')
    parser.add_argument('--gpus', default=None, type=int, nargs='+')
    parser.add_argument('--num_calibrations', default=5, type=int)
    parser.add_argument('--model_scale', default='small')
    parser.add_argument('--recalibrate', default=False)


    args = parser.parse_args()

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpus))
    main(protein_list=args.protein_list, model_type=args.model_type,
         num_calibrations=args.num_calibrations,model_size_flag=args.model_scale,
         recalibrate=args.recalibrate)