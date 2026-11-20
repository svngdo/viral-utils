import { createBrowserRouter } from "react-router";
import { SystemsPage } from "@/pages/systems/SystemsPage";

export const router = createBrowserRouter([
  {
    path: "/systems",
    children: [{ index: true, Component: SystemsPage }],
  },
]);
