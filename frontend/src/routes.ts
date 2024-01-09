import { Routes } from "@angular/router"
import { TestComponent } from "./app/testComponent/test.component"
import { AppComponent } from "./app/app.component"
import { LandingPageComponent } from "./app/landingPage/landingPage.component"


export const appRoutes:Routes = [
  { path: "", component: LandingPageComponent},
  { path: "test", component: TestComponent },

  //{ path: "", redirectTo: "/", pathMatch: 'full' },
]