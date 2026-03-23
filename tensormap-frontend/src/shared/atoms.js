import { atom, selector } from "recoil";
import * as strings from "../constants/Strings";

/**
 * Recoil atom holding the list of validated model descriptors.
 *
 * @type {import("recoil").RecoilState<Array<{ label: string, value: string, key: number }>>}
 */
export const models = atom({
  key: "models",
  default: [],
});

/**
 * Recoil atom holding the currently active project object (or null).
 *
 * @type {import("recoil").RecoilState<object | null>}
 */
export const currentProject = atom({
  key: "currentProject",
  default: null,
});

/**
 * Recoil atom holding the file list for the current project.
 *
 * @type {import("recoil").RecoilState<Array>}
 */
export const projectFiles = atom({
  key: "projectFiles",
  default: [],
});

/**
 * Recoil atom holding the full training history list.
 *
 * @type {import("recoil").RecoilState<Array>}
 */
export const trainingHistory = atom({
  key: "trainingHistory",
  default: [],
});

/**
 * Recoil selector deriving model dropdown items from training history.
 */
export const modelListSelector = selector({
  key: "modelListSelector",
  get: ({ get }) => {
    const history = get(trainingHistory);
    return history.map((m) => ({
      label: m.model_name + strings.MODEL_EXTENSION,
      value: m.model_name,
      id: m.id,
    }));
  },
});
