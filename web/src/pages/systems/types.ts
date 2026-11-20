export interface System {
  id: string;
  name: string;
  description: string | null;
  createdAt: Date;
  updatedAt: Date;
}

export type SystemList = {
  items: System[];
  total: number;
};

export type SystemFormData = {
  name: string;
  description: string;
};

export type SystemFormError = {
  name: string;
  description: string;
};

export type CreateSystemInput = SystemFormData;

export type UpdateSystemInput = { id: string } & SystemFormData;
