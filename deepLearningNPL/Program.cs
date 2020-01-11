using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Word2Vec.Net;
using Word2vec.Tools;
using NWord2Vec;

namespace deepLearningNPL
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("inserisci il path del file da confrontare:");
            string pathFile = Console.ReadLine();

            #region Word2vec.net

            string trainfile = pathFile;
            string outputFileName = "vocab.bin";

            var word2VecBuilder = Word2VecBuilder.Create();

            word2VecBuilder.WithTrainFile(trainfile);
            word2VecBuilder.WithOutputFile(outputFileName);
            word2VecBuilder.WithSize(200);
            word2VecBuilder.WithDebug(2);
            word2VecBuilder.WithBinary(1);
            word2VecBuilder.WithWindow(7);
            word2VecBuilder.WithThreads(5);
            word2VecBuilder.WithMinCount(1);
            var word2Vec = word2VecBuilder.Build();

            word2Vec.TrainModel();
            
            #endregion

            string[] lines = File.ReadAllLines(pathFile);
            List<string>[] textSplitted = new List<string>[lines.Length];
            for (int i = 0; i < lines.Length; i++)
            {
                textSplitted[i] = new List<string>();
                textSplitted[i] = (lines[i].ToLower().Split(' ')).ToList<string>();
            }

            #region Word2vec.tools
            var model = new Word2VecBinaryReader().Read("vocab.bin");

            foreach (var word in model.Words)
            {
                Console.WriteLine(word.WordOrNull);
            }


            while (true)
            {
                Console.WriteLine("Inserisci la frase da controllare: \n");
                string phraseInput = Console.ReadLine();
                string[] phraseSplitedInput = phraseInput.ToLower().Split(' ');

                if (phraseSplitedInput[0] == "stop")
                {
                    break;
                }
                double distValSum = 0;
                //l'addizione torna 1 se è la stessa parola
                for (int i = 0; i < phraseSplitedInput.Length; i++)
                {
                    for (int j = 0; j < textSplitted.Length; j++)
                    {
                        for (int z = 0; z < textSplitted[j].Count; z++)
                        {
                            var Representation = model[phraseSplitedInput[i]].Add(model[textSplitted[j][z]]);
                            var closestAdditions = model.Distance(Representation, 1);

                            if (closestAdditions[0].DistanceValue > 0.85f)
                            {
                                Console.WriteLine(closestAdditions[0].Representation.WordOrNull + "\t\t" + closestAdditions[0].DistanceValue);
                                distValSum += closestAdditions[0].DistanceValue;
                            }
                        }
                    }
                }
                Console.WriteLine("totalDist: " + distValSum);
                Console.WriteLine("AverageDist: " + (distValSum / phraseSplitedInput.Length));

                #endregion
            }
        }
    }
}
