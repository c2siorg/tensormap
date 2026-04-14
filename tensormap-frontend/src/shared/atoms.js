import { atom } from "recoil";

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
