import { atom } from "recoil";

export const models = atom({
  key: "models",
  default: [],
});

export const currentProject = atom({
  key: "currentProject",
  default: null,
});

export const projectFiles = atom({
  key: "projectFiles",
  default: [],
});
