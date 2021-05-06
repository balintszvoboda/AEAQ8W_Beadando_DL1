using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace AEAQ8W_Beadando_DL1
{
    public class DataSet
    {
        public const int InputSize = 5;
        public List<float> Input { get; set; } = new List<float>();

        public const int OutputSize = 6;
        public List<float> Output { get; set; } = new List<float>();

        public int Count { get; set; }

        public DataSet(string filename)
        {
            LoadData(filename);
        }

        void LoadData(string filename)
        {
            Count = 0;
            foreach (String line in File.ReadAllLines(filename))
            {
                var data = Normalize(line.Split('\t').Select(x => float.Parse(x)).ToList());
                Input.AddRange(data.GetRange(0, InputSize));
                for (int i = 0; i <= 5; i++)
                {
                    Output.Add(data[InputSize] == i ? 1.0f : 0.0f);
                }
                Count++;
            }
        }

        public void Shuffle()
        {
            Random rnd = new Random();
            for (int swapI = 0; swapI < Count; swapI++)
            {
                var a = rnd.Next(Count);
                var b = rnd.Next(Count);
                if (a != b)
                {
                    float T;
                    for (int i = 0; i < InputSize; i++)
                    {
                        T = Input[a * InputSize + i];
                        Input[a * InputSize + i] = Input[b * InputSize + i];
                        Input[b * InputSize + i] = T;
                    }
                    for (int i = 0; i < OutputSize; i++)
                    {
                        T = Output[a * OutputSize + i];
                        Output[a * OutputSize + i] = Output[b * OutputSize + i];
                        Output[b * OutputSize + i] = T;
                    }
                }
            }
        }

        static float[] minValues;
        static float[] maxValues;
        public static List<float> Normalize(List<float> floats)
        {
            List<float> normalized = new List<float>();
            for (int i = 0; i < floats.Count-1; i++)
                normalized.Add((floats[i] - minValues[i]) / (maxValues[i] - minValues[i]));
            normalized.Add(floats.Last());
            return normalized;
        }

        public static void LoadMinMax(string filename)
        {
            foreach (String line in File.ReadAllLines(filename))
            {
                var floats = line.Split('\t').Select(x => float.Parse(x)).ToList();
                if (minValues == null)
                {
                    minValues = floats.ToArray();
                    maxValues = floats.ToArray();
                }
                else
                {
                    for (int i = 0; i < floats.Count; i++)
                        if (floats[i] < minValues[i])
                            minValues[i] = floats[i];
                        else
                            if (floats[i] > maxValues[i])
                            maxValues[i] = floats[i];
                }
            }
        }

        public String DataToString(int index)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(index+".\nInput: [");
            for (int d = 0; d <= 4; d++)
            {
                if (d != 0) sb.Append(", ");
                sb.Append(String.Format("{0:0.00}", Input[index * InputSize + d]));
            }
            sb.Append("]\tOutput: [");
            for (int d = 0; d <= 5; d++)
            {
                if (d != 0) sb.Append(", ");
                sb.Append(String.Format("{0:0.00}", Output[index * OutputSize + d]));
            }
            sb.Append("]\n");
            return sb.ToString();
        }
    }
}
