import { defaultModels, FileUtils, InferenceSession } from '../backend';
import invariant from 'tiny-invariant';
import { ModelBase } from './ModelBase';
export class Recognition extends ModelBase {
  #dictionary;
  #accuracyMean;
  constructor(options, dictionary) {
    super(options);
    this.#dictionary = dictionary;
    this.#accuracyMean = options.options.accuracyMean ?? 0.5;
  }
  static async create({ models, onnxOptions = {}, ...restOptions }) {
    const recognitionPath =
      models?.recognitionPath || defaultModels?.recognitionPath;
    invariant(recognitionPath, 'recognitionPath is required');
    const dictionaryPath =
      models?.dictionaryPath || defaultModels?.dictionaryPath;
    invariant(dictionaryPath, 'dictionaryPath is required');
    const model = await InferenceSession.create(recognitionPath, onnxOptions);
    const dictionaryText = await FileUtils.read(dictionaryPath);
    const dictionary = [...dictionaryText.split('\n'), ' '];
    return new Recognition({ model, options: restOptions }, dictionary);
  }
  async run(lineImages, { onnxOptions = {} } = {}, fileName) {
    const modelDatas = await Promise.all(
      // Detect text from each line image
      lineImages.map(async (lineImage, index) => {
        // Resize Image to 48px height
        //  - height must <= 48
        //  - height: 48 is more accurate then 40, but same as 30
        const image = await lineImage.image.resize({
          height: 48,
        });

        // this.debugImage(lineImage.image, `./output/out9-line-${index}.jpg`)
        this.debugImage(
          lineImage.image,
          `before-rec__${fileName}__[${index}]__${Date.now()}_${String(
            Math.floor(Math.random() * 1_000_000) + 1,
          ).padStart(7, '0')}.jpg`,
        );
        // this.debugImage(image, `./output/out9-line-${index}-resized.jpg`)
        // transform image data to model data
        const modelData = this.imageToInput(image, {
          mean: [0.5, 0.5, 0.5],
          std: [0.5, 0.5, 0.5],
        });
        return modelData;
      }),
    );
    const allLines = [];
    // console.time('Recognition')
    for (const modelData of modelDatas) {
      // Run model for each line image
      const output = await this.runModel({ modelData, onnxOptions });
      // use Dictoinary to decode output to text
      const lines = await this.decodeText(output);
      allLines.unshift(...lines);
    }
    // console.timeEnd('Recognition')
    return calculateBox(
      { lines: allLines, lineImages },
      { accuracyMean: this.#accuracyMean },
    );
  }
  decodeText(output) {
    const data = output;
    const predLen = data.dims[2];
    const line = [];
    let ml = data.dims[0] - 1;
    for (let l = 0; l < data.data.length; l += predLen * data.dims[1]) {
      const predsIdx = [];
      const predsProb = [];
      for (let i = l; i < l + predLen * data.dims[1]; i += predLen) {
        const tmpArr = data.data.slice(i, i + predLen);
        const tmpMax = tmpArr.reduce(
          (a, b) => Math.max(a, b),
          Number.NEGATIVE_INFINITY,
        );
        const tmpIdx = tmpArr.indexOf(tmpMax);
        predsProb.push(tmpMax);
        predsIdx.push(tmpIdx);
      }
      line[ml] = decode(this.#dictionary, predsIdx, predsProb, true);
      ml--;
    }
    return line.filter((item) => {
      const text = item.text;
      if (text.length <= 2 && !/\d/.test(text)) {
        return false;
      }
      // Ignore lines that start with '#' or '@'
      if (text.startsWith('#') || text.startsWith('@')) {
        return false;
      }
      return true;
    });
  }
}
function decode(dictionary, textIndex, textProb, isRemoveDuplicate) {
  const ignoredTokens = [0];
  const charList = [];
  const confList = [];
  for (let idx = 0; idx < textIndex.length; idx++) {
    if (textIndex[idx] in ignoredTokens) {
      continue;
    }
    if (isRemoveDuplicate) {
      if (idx > 0 && textIndex[idx - 1] === textIndex[idx]) {
        continue;
      }
    }
    if (textIndex[idx] === 18384) {
      charList.push(' ');
    } else {
      charList.push(dictionary[textIndex[idx] - 1]);
    }
    if (textProb) {
      confList.push(textProb[idx]);
    } else {
      confList.push(1);
    }
  }
  let text = '';
  let mean = 0;
  if (charList.length) {
    text = charList.join('');
    let sum = 0;
    confList.forEach((item) => {
      sum += item;
    });
    mean = sum / confList.length;
  }
  return { text, mean };
}
function calculateBox({ lines, lineImages }, { accuracyMean }) {
  let mainLine = lines;
  const box = lineImages;
  for (const i in mainLine) {
    const b = box[mainLine.length - Number(i) - 1].box;
    for (const p of b) {
      p[0] = p[0];
      p[1] = p[1];
    }
    mainLine[i]['box'] = b;
  }
  mainLine = mainLine.filter((x) => x.mean >= accuracyMean);
  mainLine = afAfRec(mainLine);
  return mainLine;
}
function afAfRec(l) {
  const line = [];
  const ind = new Map();
  for (const i in l) {
    let item = l[i].box;
    ind.set(item, Number(i));
  }
  function calculateAverageHeight(boxes) {
    let totalHeight = 0;
    for (const box of boxes) {
      const [[, y1], , [, y2]] = box;
      const height = y2 - y1;
      totalHeight += height;
    }
    return totalHeight / boxes.length;
  }
  function groupBoxesByMidlineDifference(boxes) {
    const averageHeight = calculateAverageHeight(boxes);
    const result = [];
    for (const box of boxes) {
      const [[, y1], , [, y2]] = box;
      const midline = (y1 + y2) / 2;
      const group = result.find((b) => {
        const [[, groupY1], , [, groupY2]] = b[0];
        const groupMidline = (groupY1 + groupY2) / 2;
        return Math.abs(groupMidline - midline) < averageHeight / 2;
      });
      if (group) {
        group.push(box);
      } else {
        result.push([box]);
      }
    }
    for (const group of result) {
      group.sort((a, b) => {
        const [ltA] = a;
        const [ltB] = b;
        return ltA[0] - ltB[0];
      });
    }
    result.sort((a, b) => a[0][0][1] - b[0][0][1]);
    return result;
  }
  const boxes = groupBoxesByMidlineDifference([...ind.keys()]);
  for (const i of boxes) {
    if (i.length === 0) continue; // Skip empty arrays
    const texts = [];
    let meanSum = 0;
    for (const j of i) {
      const index = ind.get(j);
      if (index === undefined) {
        console.warn('Missing index for box:', j);
        continue;
      }
      const x = l[index];
      texts.push(x.text);
      meanSum += x.mean;
    }
    if (texts.length > 0) {
      const firstBox = i[0];
      const lastBox = i[i.length - 1];
      if (!firstBox || !lastBox) {
        console.warn('Invalid box structure');
        continue;
      }
      line.push({
        mean: meanSum / i.length,
        text: texts.join(' '),
        box: [firstBox[0], lastBox[1], lastBox[2], firstBox[3]],
      });
    }
  }
  return line;
}
//# sourceMappingURL=Recognition.js.map
